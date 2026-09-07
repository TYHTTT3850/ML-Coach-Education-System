from __future__ import annotations


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    public_message = "服务器暂时无法处理请求"

    def __init__(self, message: str | None = None, *, internal_message: str | None = None) -> None:
        super().__init__(internal_message or message or self.public_message)
        self.message = message or self.public_message
        self.internal_message = internal_message or self.message


class ConfigurationError(AppError):
    status_code = 503
    code = "service_not_configured"
    public_message = "服务尚未完成配置"


class DocumentProcessingError(AppError):
    status_code = 422
    code = "document_processing_error"
    public_message = "文档处理失败，请检查文件内容"


class UnsupportedFileError(AppError):
    status_code = 415
    code = "unsupported_file"
    public_message = "不支持该文件格式或 MIME 类型"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"
    public_message = "上传文件超过大小限制"


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"
    public_message = "该课程中已存在相同文件"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"
    public_message = "未找到该文档"


class VectorStoreError(AppError):
    status_code = 503
    code = "VECTOR_STORE_UNAVAILABLE"
    public_message = "知识库服务暂时不可用"


class EmbeddingServiceError(AppError):
    status_code = 502
    code = "embedding_service_error"
    public_message = "向量化服务暂时不可用"


class LLMServiceError(AppError):
    status_code = 502
    code = "LLM_SERVICE_UNAVAILABLE"
    public_message = "大模型服务暂时不可用，请稍后重试"


class StudentNotFoundError(AppError):
    status_code = 404
    code = "STUDENT_NOT_FOUND"
    public_message = "未找到该匿名学生，请重新进入学习页面"


class TopicNotFoundError(AppError):
    status_code = 404
    code = "TOPIC_NOT_FOUND"
    public_message = "未找到该课程知识点"


class CourseTopicMismatchError(AppError):
    status_code = 422
    code = "COURSE_TOPIC_MISMATCH"
    public_message = "知识点不属于当前课程"


class TopicLockedError(AppError):
    status_code = 409
    code = "TOPIC_LOCKED"
    public_message = "该知识点的前置要求尚未完成"


class KnowledgeInsufficientError(AppError):
    status_code = 422
    code = "KNOWLEDGE_INSUFFICIENT"
    public_message = "当前知识库中没有足够资料生成该知识点的可靠练习，请补充课程材料。"


class PracticeSetNotFoundError(AppError):
    status_code = 404
    code = "PRACTICE_SET_NOT_FOUND"
    public_message = "未找到该练习集合"


class PracticeSetAlreadySubmittedError(AppError):
    status_code = 409
    code = "PRACTICE_SET_ALREADY_SUBMITTED"
    public_message = "该练习集合已经提交"


class InvalidQuestionAnswerError(AppError):
    status_code = 422
    code = "INVALID_QUESTION_ANSWER"
    public_message = "提交的题目答案无效"


class EvaluationUncertainError(AppError):
    status_code = 422
    code = "EVALUATION_UNCERTAIN"
    public_message = "当前评价证据不足或结果不可靠，请重新提交或生成练习"


class StructuredOutputInvalidError(AppError):
    status_code = 502
    code = "LLM_STRUCTURED_OUTPUT_INVALID"
    public_message = "模型返回的结构化结果无效，请稍后重试"


class DuplicateSubmissionError(AppError):
    status_code = 409
    code = "DUPLICATE_SUBMISSION"
    public_message = "该提交正在处理或已经处理"
