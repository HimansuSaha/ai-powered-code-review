from fastapi import APIRouter, Request, HTTPException, status
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/github")
async def handle_github_webhook(request: Request):
    """Handle GitHub webhook events"""
    try:
        payload = await request.json()
        event_type = request.headers.get("X-GitHub-Event")
        
        logger.info(f"Received GitHub webhook: {event_type}")
        
        if event_type == "push":
            # Handle push event - trigger code analysis
            return await handle_push_event(payload)
        elif event_type == "pull_request":
            # Handle PR event - analyze PR changes
            return await handle_pull_request_event(payload)
        
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )

@router.post("/gitlab")
async def handle_gitlab_webhook(request: Request):
    """Handle GitLab webhook events"""
    try:
        payload = await request.json()
        event_type = request.headers.get("X-Gitlab-Event")
        
        logger.info(f"Received GitLab webhook: {event_type}")
        
        return {"status": "received"}
        
    except Exception as e:
        logger.error(f"GitLab webhook processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )

async def handle_push_event(payload):
    """Handle GitHub push event"""
    # This would typically trigger background analysis
    repository = payload.get("repository", {})
    commits = payload.get("commits", [])
    
    logger.info(f"Push to {repository.get('full_name')}: {len(commits)} commits")
    
    # TODO: Queue background analysis tasks
    return {"message": "Push event processed"}

async def handle_pull_request_event(payload):
    """Handle GitHub pull request event"""
    action = payload.get("action")
    pull_request = payload.get("pull_request", {})
    
    logger.info(f"PR {action}: #{pull_request.get('number')}")
    
    # TODO: Analyze PR changes
    return {"message": "PR event processed"}