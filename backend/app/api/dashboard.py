from fastapi import APIRouter, Depends
from ..api.auth import get_current_user
from ..models.user import User

router = APIRouter()

@router.get("/stats")
async def get_dashboard_stats(current_user: User = Depends(get_current_user)):
    """Get dashboard statistics"""
    return {
        "total_analyses": 156,
        "security_issues": 23,
        "quality_issues": 45,
        "average_score": 78.5,
        "recent_analyses": [
            {
                "id": "1",
                "file_path": "src/main.py",
                "score": 85.2,
                "issues": 3,
                "timestamp": "2024-01-01T10:00:00Z"
            },
            {
                "id": "2", 
                "file_path": "src/utils.py",
                "score": 92.1,
                "issues": 1,
                "timestamp": "2024-01-01T09:30:00Z"
            }
        ]
    }

@router.get("/trends")
async def get_analysis_trends(current_user: User = Depends(get_current_user)):
    """Get analysis trends over time"""
    return {
        "score_trend": [
            {"date": "2024-01-01", "score": 78.5},
            {"date": "2024-01-02", "score": 80.1},
            {"date": "2024-01-03", "score": 82.3}
        ],
        "issues_trend": [
            {"date": "2024-01-01", "critical": 2, "high": 5, "medium": 8, "low": 12},
            {"date": "2024-01-02", "critical": 1, "high": 4, "medium": 7, "low": 10},
            {"date": "2024-01-03", "critical": 1, "high": 3, "medium": 6, "low": 8}
        ]
    }