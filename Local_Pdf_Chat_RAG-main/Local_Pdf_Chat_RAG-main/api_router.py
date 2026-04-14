"""
REST API 模块（使用FastAPI实现）
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import re
from typing import Dict, Any, List, Optional
import logging
import asyncio
from contextlib import asynccontextmanager

# 从重构后的模块导入
from config import SILICONFLOW_API_KEY
from core.generator import query_answer
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from features.web_search import check_serpapi_key
from utils.network import is_port_available

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-api")
INDEX_STORAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "indexes", "default"
)


class ProgressCallback:
    def __init__(self):
        self.progress = 0
        self.description = ""

    def __call__(self, progress, desc=None):
        self.progress = progress
        self.description = desc or ""
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API 服务启动")
    loaded, message = vector_store.load_index(INDEX_STORAGE_PATH)
    if loaded:
        bm25_manager.load(INDEX_STORAGE_PATH)
        logger.info("启动加载索引成功: %s", message)
    else:
        logger.info("启动未加载索引: %s", message)
    yield
    logger.info("API 服务已关闭")


app = FastAPI(
    title="本地RAG API服务",
    description="提供基于本地大模型和SERPAPI的文档问答API接口",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    enable_web_search: bool = False
    model_choice: str = "siliconflow"


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class FileProcessResult(BaseModel):
    status: str
    message: str
    file_info: Optional[Dict[str, Any]] = None


@app.post("/api/upload", response_model=FileProcessResult)
async def upload_file(file: UploadFile = File(...), index_mode: str = Form("append")):
    """处理文档并存入向量数据库"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from rag_demo import process_multiple_files
        progress = ProgressCallback()

        result_text = await asyncio.to_thread(
            process_multiple_files,
            [type('obj', (object,), {"name": tmp_path})],
            index_mode,
            progress
        )

        os.unlink(tmp_path)
        result = result_text[0] if isinstance(result_text, tuple) else result_text
        chunk_match = re.search(r'(\d+) 个文本块', result)
        chunks = int(chunk_match.group(1)) if chunk_match else 0

        return {
            "status": "success" if "成功" in result else "error",
            "message": result,
            "file_info": {"filename": file.filename, "chunks": chunks}
        }
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        raise HTTPException(500, f"文档处理失败: {str(e)}") from e


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """问答接口"""
    if not req.question:
        raise HTTPException(400, "问题不能为空")
    try:
        answer = await asyncio.to_thread(query_answer, req.question, req.enable_web_search, req.model_choice)
        sources = []
        url_matches = re.findall(r'\[(网络来源|本地文档):[^\]]+\]\s*(?:\(URL:\s*([^)]+)\))?', answer)
        for source_type, url in url_matches:
            sources.append({"type": source_type, "url": url} if url else {"type": source_type})

        return {
            "answer": answer, "sources": sources,
            "metadata": {"enable_web_search": req.enable_web_search, "model": req.model_choice}
        }
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(500, f"问答处理失败: {str(e)}") from e


@app.get("/api/status")
async def check_status():
    return {
        "status": "healthy",
        "siliconflow_configured": bool(SILICONFLOW_API_KEY and not SILICONFLOW_API_KEY.startswith("Your")),
        "serpapi_configured": check_serpapi_key(),
        "vector_store_ready": vector_store.is_ready,
        "total_chunks": vector_store.total_chunks,
        "version": "2.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    port = next((p for p in [17995, 17996, 17997, 17998, 17999] if is_port_available(p)), 17995)
    logger.info(f"启动API服务，端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)