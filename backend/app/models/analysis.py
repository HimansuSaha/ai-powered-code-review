from sqlalchemy import Column, String, DateTime, Text, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..core.database import Base

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=True)
    file_path = Column(String, nullable=False)
    language = Column(String, nullable=False)
    analysis_type = Column(String, nullable=False)  # full, security, quality
    overall_score = Column(Float, nullable=False, default=0.0)
    
    # JSON fields for storing analysis results
    issues = Column(JSON, nullable=True, default=list)
    metrics = Column(JSON, nullable=True, default=dict)
    suggestions = Column(JSON, nullable=True, default=list)
    
    # Optional fields for git integration
    commit_sha = Column(String, nullable=True)
    branch = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Analysis(file_path='{self.file_path}', score={self.overall_score})>"

class AnalysisIssue(Base):
    __tablename__ = "analysis_issues"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False)
    
    issue_type = Column(String, nullable=False)  # vulnerability_type or code_smell_type
    category = Column(String, nullable=False)  # security, quality, performance
    severity = Column(String, nullable=False)  # critical, high, medium, low, info
    message = Column(Text, nullable=False)
    
    # Location in code
    line_number = Column(String, nullable=True)
    column_number = Column(String, nullable=True)
    
    # Additional context
    confidence_score = Column(Float, nullable=True)
    suggested_fix = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Analysis CRUD operations
async def create_analysis(
    db: AsyncSession,
    user_id: uuid.UUID,
    file_path: str,
    language: str,
    analysis_type: str,
    overall_score: float,
    issues: List[Dict[str, Any]] = None,
    metrics: Dict[str, Any] = None,
    suggestions: List[Dict[str, Any]] = None,
    repository_id: Optional[uuid.UUID] = None,
    commit_sha: Optional[str] = None,
    branch: Optional[str] = None
) -> Analysis:
    """Create a new analysis record"""
    
    analysis = Analysis(
        user_id=user_id,
        repository_id=repository_id,
        file_path=file_path,
        language=language,
        analysis_type=analysis_type,
        overall_score=overall_score,
        issues=issues or [],
        metrics=metrics or {},
        suggestions=suggestions or [],
        commit_sha=commit_sha,
        branch=branch
    )
    
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)
    return analysis

async def get_analysis_by_id(db: AsyncSession, analysis_id: uuid.UUID) -> Optional[Analysis]:
    """Get analysis by ID"""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    return result.scalar_one_or_none()

async def get_user_analyses(
    db: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0
) -> List[Analysis]:
    """Get analyses for a user"""
    result = await db.execute(
        select(Analysis)
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()

async def get_repository_analyses(
    db: AsyncSession,
    repository_id: uuid.UUID,
    limit: int = 50,
    offset: int = 0
) -> List[Analysis]:
    """Get analyses for a repository"""
    result = await db.execute(
        select(Analysis)
        .where(Analysis.repository_id == repository_id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()

async def delete_analysis(db: AsyncSession, analysis_id: uuid.UUID) -> bool:
    """Delete an analysis"""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    
    if analysis:
        await db.delete(analysis)
        await db.commit()
        return True
    return False

async def get_analysis_statistics(db: AsyncSession, user_id: uuid.UUID) -> Dict[str, Any]:
    """Get analysis statistics for a user"""
    # This would typically involve more complex queries
    # For now, return basic stats
    result = await db.execute(
        select(Analysis).where(Analysis.user_id == user_id)
    )
    analyses = result.scalars().all()
    
    if not analyses:
        return {
            "total_analyses": 0,
            "average_score": 0,
            "total_issues": 0,
            "languages": []
        }
    
    total_issues = sum(len(analysis.issues) for analysis in analyses)
    average_score = sum(analysis.overall_score for analysis in analyses) / len(analyses)
    languages = list(set(analysis.language for analysis in analyses))
    
    return {
        "total_analyses": len(analyses),
        "average_score": round(average_score, 2),
        "total_issues": total_issues,
        "languages": languages
    }