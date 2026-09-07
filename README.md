# 启动方式

## 环境变量

复制 `backend/.env.example` 为 `backend/.env.local`，至少配置模型服务：

```env
LLM_BASE_URL=https://your-llm-provider.example/v1
LLM_API_KEY=your-llm-api-key
LLM_MODEL=your-chat-model

EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_MODEL=your-embedding-model
```

复制 `frontend/.env.example` 为 `frontend/.env.local`，配置：

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

如果前后端不在同一台机器或端口，需要同步修改后端的 `FRONTEND_ORIGIN` 和前端的 `NEXT_PUBLIC_API_BASE_URL` 。

## 后端

Windows PowerShell：

```powershell
cd backend
./.venv/Scripts/python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端健康检查：`http://localhost:8000/health`

## 前端

```powershell
cd frontend
npm run dev
```

然后访问 `http://localhost:3000`。

## 测试和构建

```powershell
# 后端测试
cd backend
./.venv/Scripts/python.exe -m pytest -q

# 前端检查和生产构建
cd ../frontend
npm run typecheck
npm run lint
npm run build
```

## 运行数据说明

`backend/data/` 下的 SQLite 数据库、Chroma 索引和上传文件属于运行时数据。删除它们会清除本地学生学习记录、文档索引或上传内容；重新启动后，课程目录会由内置 seed 数据幂等初始化。

# 项目结构

```text
ML-Coach-Education-System/
├── README.md                         # 项目说明、安装和启动文档
├── AI赋能教育_ML-Coach_Agent方案.md    # 产品方案、系统设计和 Agent 思路
├── 机器学习方法/                       # 机器学习课程原始 Markdown 教材
├── backend/                          # FastAPI 后端
│   ├── app/                          # 后端应用源码
│   ├── data/                         # 运行时数据库、向量库和上传文件
│   ├── knowledge/                    # 用于构建知识库的课程资料
│   ├── scripts/                      # 内容审计等辅助脚本
│   ├── tests/                        # 后端自动化测试
│   ├── requirements.txt              # Python 依赖
│   └── .env.example                  # 后端环境变量模板
└── frontend/                         # Next.js 前端
	├── app/                          # App Router 页面和路由
	├── components/                   # 可复用 React 组件
	├── lib/                          # API、类型、状态和业务辅助函数
	├── public/                       # 静态资源
	├── scripts/                      # 浏览器冒烟测试等脚本
	├── package.json                  # 前端依赖和 npm 命令
	├── next.config.mjs               # Next.js 配置
	└── .env.example                  # 前端环境变量模板
```

## 根目录文件和课程资料

- `README.md`：项目结构、环境配置、启动方式和测试命令。
- `AI赋能教育_ML-Coach_Agent方案.md`：系统定位、功能规划和智能辅导 Agent 方案。
- `机器学习方法/00-目录与前言.md`：机器学习教材目录和前言。
- `机器学习方法/机器学习方法-MOC.md`：课程内容的知识地图和导航入口。
- `机器学习方法/篇/`：按篇组织的课程目录文件。
- `机器学习方法/chapters/`：具体章节教材，从感知机、决策树、支持向量机到神经网络、预训练语言模型等。
- `机器学习方法/images/`：教材中引用的图片资源。

## 后端目录和文件

后端使用 FastAPI，启动入口是 `backend/app/main.py`，默认监听 `8000` 端口。

### `backend/app/`

- `main.py`：创建 FastAPI 应用，注册路由、中间件、异常处理、生命周期初始化和 CORS。
- `__init__.py`：定义后端应用包及版本信息。

### `backend/app/api/`

API 路由层只负责 HTTP 请求、参数接收和响应组织，具体业务委托给服务层。

- `chat.py`：课程问答接口。
- `content.py`：课程内容和内容评估相关接口。
- `course_content.py`：课程章节内容、主题内容和覆盖情况接口。
- `curriculum.py`：课程主题、阶段、前置关系和学习状态接口。
- `dependencies.py`：API 公共依赖，例如当前学生、管理员权限和应用状态。
- `health.py`：健康检查接口 `/health`，报告 LLM、Embedding 和向量库状态。
- `knowledge.py`：知识文档上传、导入、查询和管理接口。
- `learning.py`：学习仪表盘、学习主题和学习记录接口。
- `practice.py`：练习集、练习题和作答提交接口。
- `students.py`：匿名学生创建和学生信息接口。

### `backend/app/core/`

全局配置、基础类型和横切能力。

- `config.py`：读取 `.env.local`，定义模型服务、数据库、向量库、RAG、练习和掌握度配置。
- `exceptions.py`：应用级异常和错误码。
- `logging.py`：日志格式、日志级别和敏感信息脱敏配置。
- `security.py`：管理员令牌等安全校验。
- `types.py`：课程、学习状态等跨模块共享类型。

### `backend/app/content/`

课程内容导入和内容种子数据。

- `importing.py`：从 Markdown、PDF、DOCX 等来源导入课程内容。
- `seeds/`：内置课程内容、主题内容和示例数据。

### `backend/app/curriculum/`

课程知识图谱和学习路径。

- `models.py`：课程主题、阶段、前置关系等领域模型。
- `seed.py`：初始化和幂等写入课程目录。
- `seeds/`：机器学习、高等数学等课程的目录种子数据。
- `validation.py`：校验课程目录、主题关系和前置依赖。

### `backend/app/learning/`

学习过程中的掌握度、门禁和教学编排逻辑。

- `gate.py`：前置知识和掌握度门禁判断。
- `mastery.py`：根据练习证据计算主题和维度掌握度。
- `misconceptions.py`：识别和整理学生的常见错误或误解。
- `orchestrator.py`：编排学习、练习、评价和补救流程。

### `backend/app/rag/`

知识库检索增强生成（RAG）流水线。

- `bm25.py`：基于 BM25 的关键词检索。
- `chunker.py`：将课程文档切分为适合检索的文本块。
- `hyde.py`：生成假设性文档，用于 HyDE 检索增强。
- `loaders.py`：加载 Markdown、PDF、DOCX 等文件。
- `pipeline.py`：串联加载、切分、Embedding、入库和检索流程。
- `prompt_builder.py`：根据检索结果构造问答 Prompt。
- `retriever.py`：融合向量检索和 BM25 的混合检索器。
- `vector_store.py`：Chroma 向量库的初始化、写入、查询和删除。

### `backend/app/db/`

SQLite 数据库和持久化结构。

- `database.py`：数据库连接、初始化和基础数据库操作。
- `migrations.py`：数据库表结构迁移。
- `content_schema.py`：课程内容相关数据表结构。
- `content_curriculum.py`：课程内容与课程目录之间的数据库操作。

### `backend/app/schemas/`

Pydantic 请求和响应模型，用于 API 数据校验与序列化。

- `chat.py`：问答请求和响应模型。
- `content.py`：课程内容模型。
- `curriculum.py`：课程目录、主题和前置关系模型。
- `health.py`：健康检查响应模型。
- `knowledge.py`：知识文档和上传结果模型。
- `learning.py`：学习仪表盘、学习记录和掌握度模型。
- `practice.py`：练习集、题目、作答和评价模型。
- `practice_api.py`：练习 API 的输入输出模型。
- `student.py`：学生信息模型。

### `backend/app/services/`

业务服务层，负责协调数据库、模型服务、RAG 和学习领域逻辑。

- `assessment_service.py`：课程内容和学习结果的评估。
- `content_repository.py`：课程内容数据访问。
- `document_registry.py`：知识文档注册、状态和元数据管理。
- `evaluation_service.py`：练习答案和开放题评价。
- `knowledge_service.py`：知识文档导入、切分、Embedding 和索引服务。
- `learning_grounding.py`：为学习流程提供课程和学生状态上下文。
- `learning_repository.py`：学生、学习记录、练习和掌握度的持久化。
- `learning_service.py`：学习流程的业务服务。
- `practice_service.py`：练习生成、组卷和提交处理。
- `remediation_service.py`：根据评价结果生成补救学习建议。
- `structured_output.py`：解析和校验 LLM 的结构化输出。
- `embedding/`：Embedding 模型服务适配器。
- `llm/`：LLM 服务适配器和不同 API 风格的调用实现。

### `backend/app/prompts/`

不同课程和学习任务使用的 Prompt 模板，包括机器学习、高等数学、问答、练习和学习流程 Prompt。

### `backend/app/cli/`

命令行维护工具。

- `ingest.py`：导入课程文档并构建知识库。
- `import_question_bank.py`：导入题库。
- `migrate.py`：执行数据库迁移。
- `seed_curriculum.py`：初始化课程目录。
- `seed_learning_content.py`：初始化课程学习内容。
- `validate_content.py`：验证课程内容完整性和格式。

### `backend/data/`、`backend/knowledge/`、`backend/scripts/`

- `data/chroma/`：Chroma 持久化向量库，由运行时生成，通常不手工编辑。
- `data/uploads/`：用户上传的文档。
- `knowledge/advanced_mathematics/`：高等数学知识库原始内容。
- `knowledge/machine_learning/`：机器学习知识库原始内容。
- `scripts/audit_content_upgrade.py`：审计课程内容升级情况。

### `backend/tests/`

后端测试目录，可以对配置、API、文档加载、切分、BM25、混合检索、HyDE、课程目录、练习生成、答案评价、学习闭环和掌握度等功能进行自动化测试。

- `conftest.py`：pytest fixtures 和测试应用配置。
- `mocks.py`：LLM、Embedding 等外部依赖的测试替身。
- `test_*.py`：按功能划分的单元测试、接口测试和端到端测试。

## 前端目录和文件

前端使用 Next.js App Router、React、TypeScript 和 Tailwind CSS，默认运行在 `3000` 端口。

### `frontend/app/`

- `layout.tsx`：全局布局、页面元数据、KaTeX 样式和学生状态 Provider。
- `page.tsx`：首页课程问答界面。
- `globals.css`：全局样式、颜色、排版和响应式布局。
- `knowledge/page.tsx`：知识库文档管理页面。
- `knowledge/layout.tsx`：知识库页面布局。
- `learn/page.tsx`：学习中心入口。
- `learn/layout.tsx`：学习中心布局。
- `learn/[course]/`：指定课程的学习仪表盘和主题列表。
- `learn/[course]/[topicId]/`：指定主题的学习详情、内容和练习页面。
- `question-bank/page.tsx`：题库使用页面。
- `question-bank/layout.tsx`：题库页面布局。
- `question-bank/admin/page.tsx`：题库管理页面。

### `frontend/components/`

可复用的 UI 和业务组件。

- `ChatInput.tsx`：问题输入框和发送控制。
- `ChatMessage.tsx`：单条聊天消息展示。
- `ChatWindow.tsx`：聊天消息列表和对话窗口。
- `CourseSelector.tsx`：课程选择器。
- `SourcePanel.tsx`：展示回答引用的知识来源。
- `MarkdownRenderer.tsx`：渲染 Markdown、GFM 和数学公式。
- `StudentProvider.tsx`：创建并维护匿名学生身份。
- `CourseLayout.tsx`、`LearningShell.tsx`：课程学习页面的整体布局。
- `LearningWidgets.tsx`：学习进度、掌握度和推荐动作组件。
- `LessonContent.tsx`：课程正文内容展示。
- `PracticePanel.tsx`：练习题、选项、答案提交和反馈展示。
- `TopicTutor.tsx`：主题辅导和问答组件。
- `TopicPrerequisiteCard.tsx`：主题前置知识状态展示。
- `KnowledgeUploader.tsx`：知识文档上传组件。
- `DocumentTopicTagger.tsx`：为文档标记课程主题。
- `ContentCoverageTable.tsx`：课程内容覆盖度表格。
- `Icons.tsx`：项目内复用的图标组件。

### `frontend/lib/`

- `api.ts`：统一封装后端请求、错误处理和问答、学习、练习 API。
- `api-content.ts`：课程内容和知识库 API 请求。
- `learning.ts`：学习流程相关的前端数据处理。
- `types.ts`：问答、课程、学习、练习和学生数据类型。
- `types-content.ts`：课程内容和知识库数据类型。
- `storage.ts`：浏览器本地对话状态的保存、加载和初始化。

### `frontend/public/`、`frontend/scripts/` 和配置文件

- `public/`：可直接由浏览器访问的静态资源，目前保留 `.gitkeep`。
- `scripts/browser-smoke.mjs`：浏览器端冒烟测试脚本。
- `package.json`：依赖和 `dev`、`build`、`start`、`typecheck`、`lint` 等命令。
- `package-lock.json`：锁定 npm 依赖版本。
- `next.config.mjs`：Next.js 配置，当前启用 standalone 输出。
- `tsconfig.json`：TypeScript 编译配置。
- `tailwind.config.ts`：Tailwind CSS 配置。
- `postcss.config.mjs`：PostCSS 配置。
- `eslint.config.mjs`：ESLint 配置。
- `.env.example`：前端环境变量模板。
- `.env.local`：本机前端环境变量，不应提交真实密钥。