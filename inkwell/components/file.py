from typing import Dict, Optional

from pydantic import BaseModel, Json


class File(BaseModel):
    data: bytes
    metadata: Optional[Dict[str, Json]] = None
    sha_256: Optional[str] = None
    mime_type: Optional[str] = None
