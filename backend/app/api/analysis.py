from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import uuid
from datetime import datetime

from ..core.database import get_db
from ..api.auth import get_current_user
from ..models.user import User
from ..models.analysis import Analysis, create_analysis, get_analysis_by_id
from ..ml_models.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models
class AnalyzeRequest(BaseModel):
    code_content: str
    file_path: str
    language: str
    analysis_type: str = "full"  # full, security, quality

class BatchAnalyzeRequest(BaseModel):
    files: List[Dict[str, str]]  # [{"path": "...", "content": "...", "language": "..."}]
    repository_id: Optional[str] = None
    commit_sha: Optional[str] = None
    analysis_type: str = "full"

class AnalysisResult(BaseModel):
    id: str
    file_path: str
    language: str
    analysis_type: str
    overall_score: float
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    suggestions: List[Dict[str, Any]]
    created_at: str

    class Config:
        from_attributes = True

class SupportedLanguagesResponse(BaseModel):
    languages: List[Dict[str, Any]]

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_code(
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Analyze a single code file"""
    try:
        # Get model manager from app state
        from main import app
        model_manager: ModelManager = app.state.model_manager
        
        # Perform analysis
        analysis_results = await model_manager.analyze_code(
            code_content=request.code_content,
            file_path=request.file_path,
            analysis_type=request.analysis_type
        )
        
        if "error" in analysis_results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {analysis_results['error']}"
            )
        
        # Extract issues and metrics
        issues = []
        metrics = {}
        suggestions = []
        
        # Process security results
        if "security" in analysis_results:
            security = analysis_results["security"]
            if security.get("vulnerabilities"):
                for vuln in security["vulnerabilities"]:
                    issues.append({
                        "type": vuln["type"],
                        "severity": vuln["severity"],
                        "message": vuln["description"],
                        "category": "security"
                    })
            
            metrics["vulnerability_score"] = security.get("vulnerability_score", 0)
            metrics["is_vulnerable"] = security.get("is_vulnerable", False)
        
        # Process quality results
        if "quality" in analysis_results:
            quality = analysis_results["quality"]
            if quality.get("code_smells"):
                for smell in quality["code_smells"]:
                    issues.append({
                        "type": smell["type"],
                        "severity": smell["severity"],
                        "message": smell["description"],
                        "line": smell.get("line"),
                        "category": "quality"
                    })
            
            metrics["complexity"] = quality.get("complexity", 0)
            metrics["maintainability_index"] = quality.get("maintainability_index", 0)
            metrics["lines_of_code"] = quality.get("lines_of_code", 0)
            metrics["quality_score"] = quality.get("quality_score", 0)
        
        # Generate suggestions based on issues
        suggestions = generate_suggestions(issues)
        
        # Save analysis to database
        analysis = await create_analysis(
            db=db,
            user_id=current_user.id,
            file_path=request.file_path,
            language=request.language,
            analysis_type=request.analysis_type,
            overall_score=analysis_results.get("overall_score", 0),
            issues=issues,
            metrics=metrics,
            suggestions=suggestions
        )
        
        logger.info(f"Analysis completed for {request.file_path} by user {current_user.email}")
        
        return AnalysisResult(
            id=str(analysis.id),
            file_path=analysis.file_path,
            language=analysis.language,
            analysis_type=analysis.analysis_type,
            overall_score=analysis.overall_score,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions,
            created_at=analysis.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed"
        )

@router.post("/batch-analyze")
async def batch_analyze_code(
    request: BatchAnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Analyze multiple code files"""
    try:
        from main import app
        model_manager: ModelManager = app.state.model_manager
        
        results = []
        
        for file_data in request.files:
            try:
                # Analyze each file
                analysis_results = await model_manager.analyze_code(
                    code_content=file_data["content"],
                    file_path=file_data["path"],
                    analysis_type=request.analysis_type
                )
                
                if "error" not in analysis_results:
                    # Process results (similar to single analysis)
                    issues = []
                    metrics = {}
                    
                    if "security" in analysis_results:
                        security = analysis_results["security"]
                        if security.get("vulnerabilities"):
                            for vuln in security["vulnerabilities"]:
                                issues.append({
                                    "type": vuln["type"],
                                    "severity": vuln["severity"],
                                    "message": vuln["description"],
                                    "category": "security"
                                })
                        metrics.update(security)
                    
                    if "quality" in analysis_results:
                        quality = analysis_results["quality"]
                        if quality.get("code_smells"):
                            for smell in quality["code_smells"]:
                                issues.append({
                                    "type": smell["type"],
                                    "severity": smell["severity"],
                                    "message": smell["description"],
                                    "line": smell.get("line"),
                                    "category": "quality"
                                })
                        metrics.update(quality)
                    
                    # Save to database
                    analysis = await create_analysis(
                        db=db,
                        user_id=current_user.id,
                        file_path=file_data["path"],
                        language=file_data.get("language", "unknown"),
                        analysis_type=request.analysis_type,
                        overall_score=analysis_results.get("overall_score", 0),
                        issues=issues,
                        metrics=metrics,
                        repository_id=request.repository_id,
                        commit_sha=request.commit_sha
                    )
                    
                    results.append({
                        "file_path": file_data["path"],
                        "analysis_id": str(analysis.id),
                        "overall_score": analysis_results.get("overall_score", 0),
                        "issues_count": len(issues),
                        "status": "completed"
                    })
                else:
                    results.append({
                        "file_path": file_data["path"],
                        "status": "failed",
                        "error": analysis_results["error"]
                    })
                    
            except Exception as e:
                logger.error(f"Failed to analyze {file_data['path']}: {e}")
                results.append({
                    "file_path": file_data["path"],
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "batch_id": str(uuid.uuid4()),
            "total_files": len(request.files),
            "completed": len([r for r in results if r["status"] == "completed"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch analysis failed"
        )

@router.get("/results/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_results(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get analysis results by ID"""
    try:
        analysis = await get_analysis_by_id(db, uuid.UUID(analysis_id))
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Check if user owns this analysis
        if analysis.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return AnalysisResult(
            id=str(analysis.id),
            file_path=analysis.file_path,
            language=analysis.language,
            analysis_type=analysis.analysis_type,
            overall_score=analysis.overall_score,
            issues=analysis.issues,
            metrics=analysis.metrics,
            suggestions=analysis.suggestions,
            created_at=analysis.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analysis results"
        )

@router.get("/supported-languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages():
    """Get list of supported programming languages"""
    languages = [
        {
            "name": "Python",
            "extensions": [".py"],
            "features": ["security", "quality", "complexity"]
        },
        {
            "name": "JavaScript",
            "extensions": [".js", ".jsx"],
            "features": ["security", "quality"]
        },
        {
            "name": "TypeScript",
            "extensions": [".ts", ".tsx"],
            "features": ["security", "quality"]
        },
        {
            "name": "Java",
            "extensions": [".java"],
            "features": ["security", "quality"]
        },
        {
            "name": "C#",
            "extensions": [".cs"],
            "features": ["security", "quality"]
        },
        {
            "name": "Go",
            "extensions": [".go"],
            "features": ["security", "quality"]
        }
    ]
    
    return SupportedLanguagesResponse(languages=languages)

@router.get("/models/status")
async def get_models_status(current_user: User = Depends(get_current_user)):
    """Get ML models status"""
    try:
        from main import app
        model_manager: ModelManager = app.state.model_manager
        
        status_info = await model_manager.get_models_status()
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get models status"
        )

def generate_suggestions(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate improvement suggestions based on issues"""
    suggestions = []
    
    # Security suggestions
    security_issues = [issue for issue in issues if issue["category"] == "security"]
    if security_issues:
        suggestions.append({
            "type": "security",
            "priority": "high",
            "title": "Security Improvements",
            "description": "Address security vulnerabilities to protect your application",
            "actions": [
                "Review and fix SQL injection vulnerabilities",
                "Avoid using eval() and exec() functions",
                "Validate and sanitize user inputs",
                "Use parameterized queries for database operations"
            ]
        })
    
    # Quality suggestions
    quality_issues = [issue for issue in issues if issue["category"] == "quality"]
    if quality_issues:
        suggestions.append({
            "type": "quality",
            "priority": "medium",
            "title": "Code Quality Improvements",
            "description": "Improve code maintainability and readability",
            "actions": [
                "Break down long methods into smaller functions",
                "Add meaningful comments and documentation",
                "Follow consistent naming conventions",
                "Remove TODO comments and implement fixes"
            ]
        })
    
    return suggestions