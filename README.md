# 技术栈

- 前端：Next.js、React、TypeScript、Tailwind CSS。
- 后端：FastAPI、Pydantic、Uvicorn、Python。
- AI 能力：兼容 OpenAI API 的 LLM 与 Embedding 服务，结合 BM25 和向量检索实现 RAG。
- 数据库：PostgreSQL 与 `pgvector`。业务数据、文档分块、来源元数据和 Embedding 向量均持久化在 PostgreSQL；向量存储于 `document_chunks.embedding`，不依赖独立向量数据库。

> 启动前，必须在 PostgreSQL 服务端安装与 PostgreSQL 主版本匹配的 `pgvector` 扩展文件。安装完成后，数据库初始化脚本会执行 `CREATE EXTENSION IF NOT EXISTS vector`；若扩展文件尚未安装，该命令会失败。

# 启动方式

## 环境变量

复制 `backend/.env.example` 为 `backend/.env.local`，配置数据库和前端来源：

```env
# PostgreSQL 连接串示例
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ml_coach

# 前端地址，默认即可用于本地开发
FRONTEND_ORIGIN=http://localhost:3000
```

复制 `frontend/.env.example` 为 `frontend/.env.local`，配置：

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

如果前后端不在同一台机器或端口，需要同步修改后端的 `FRONTEND_ORIGIN` 和前端的 `NEXT_PUBLIC_API_BASE_URL` 。

启动前端后，打开“设置”页，选择对话模型和 Embedding 模型厂商，并填写各自的 API Key；密钥仅保存于当前浏览器的本地存储，并会随请求发送给后端。自定义兼容服务时，也在该页面填写对应的 API Base URL。

## 运维初始化

根目录 `operations/` 存放部署、初始化和维护脚本，不会由后端运行时自动执行。首次启动后端前，在项目根目录运行：

```powershell
.\backend\.venv\Scripts\python.exe .\operations\database\initialize.py
```

脚本读取 `backend/.env.local` 中的 `DATABASE_URL`，执行 `operations/database/initialize_postgresql.sql` 创建当前全部表、索引、函数和触发器。该脚本面向空数据库；重复执行不会重复创建已有对象，但不会把旧版本数据库升级到当前结构。

## 数据备份与迁移

代码包不包含 PostgreSQL 中的学生、题库、学习记录、上传文档元数据和 Embedding 向量。需要交付现有数据时，安装 PostgreSQL 客户端工具，并在源端导出：

```powershell
$env:DATABASE_URL = "postgresql://user:password@host:5432/ml_coach"
.\operations\database\backup.ps1 -OutputPath .\ml-coach.backup
```

在接收方创建空的 PostgreSQL 数据库、安装匹配版本的 `pgvector` 后，使用同样方式设置目标 `DATABASE_URL`，再恢复备份：

```powershell
.\operations\database\restore.ps1 -BackupPath .\ml-coach.backup
```

恢复会清除目标数据库中的同名对象，目标必须是专用的可覆盖数据库。仅交付空系统时，无需备份和恢复，只执行上面的数据库初始化命令。

## PostgreSQL 架构测试

后端测试默认不会连接数据库。为验证全新 PostgreSQL 数据库的建表脚本，请创建专用空库（不可指向开发或生产库），并运行：

```powershell
$env:POSTGRES_TEST_DATABASE_URL = "postgresql://user:password@host:5432/ml_coach_test"
cd backend
.\.venv\Scripts\python.exe -m pytest -q tests/test_learning_migrations.py
```

该测试会连续两次执行初始化脚本，并校验 `pgvector` 扩展及核心表是否存在。

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

业务数据和向量均存储在 `DATABASE_URL` 指向的 PostgreSQL 数据库中；`backend/data/` 下仅保留上传文件。首次部署须在项目根目录执行 `backend/.venv/Scripts/python.exe operations/database/initialize.py`；应用运行时不会创建或修改表结构。

# 项目结构

```text
ML-Coach-Education-System/
├── README.md                         # 项目说明、安装和启动文档
├── AI赋能教育_ML-Coach_Agent方案.md    # 产品方案、系统设计和 Agent 思路
├── operations/                         # 一次性初始化与运维脚本
│   └── database/                       # PostgreSQL 初始化工具
│       ├── initialize.py               # 数据库初始化入口
│       └── initialize_postgresql.sql   # 当前完整数据库结构
├── backend/                          # FastAPI 后端
│   ├── app/                          # 后端应用源码
│   ├── data/                         # 上传文件
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
- `operations/`：项目初始化、升级和其他非运行时运维脚本。

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
- `vector_store.py`：PostgreSQL pgvector 的初始化、写入、查询和删除。

### `backend/app/db/`

PostgreSQL 运行时连接和持久化结构。

- `database.py`：数据库连接和基础数据库操作。

### `operations/database/`

一次性 PostgreSQL 数据库初始化脚本，不会由运行时应用自动调用。请从项目根目录执行初始化命令。

- `initialize.py`：执行 PostgreSQL 初始化。
- `initialize_postgresql.sql`：当前完整的建表、索引、函数和触发器 SQL。

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
- `seed_curriculum.py`：初始化课程目录。
- `seed_learning_content.py`：初始化课程学习内容。
- `validate_content.py`：验证课程内容完整性和格式。

### `backend/data/`、`backend/knowledge/`、`backend/scripts/`

- `data/uploads/`：用户上传的文档。
- `knowledge/advanced_mathematics/`：高等数学知识库原始内容。
- `knowledge/machine_learning/`：机器学习知识库原始内容。

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

# 数据库表结构

数据库结构由 `operations/database/initialize_postgresql.sql` 定义，当前使用 PostgreSQL，共 24 张业务表。`*_json` 字段和 `JSONB` 字段保存结构化 JSON 数据；时间字段当前统一按文本格式保存，由应用层写入 ISO 8601 时间。除特别说明外，`id` 为业务标识，`created_at`、`updated_at` 等字段为记录时间。

## 表清单

### `assessment_blueprints`：测评蓝图

保存生成练习或检查点时使用的组卷蓝图。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，蓝图标识 |
| `payload_json` | 蓝图内容，包括组卷规则等 |
| `created_at` | 创建时间 |

### `content_exposures`：内容曝光记录

记录学生接触课程内容或题目内容的情况，用于分析内容曝光与测评之间的关系。

| 字段 | 说明 |
| --- | --- |
| `id` | PostgreSQL identity 自增主键 |
| `student_id` | 学生 |
| `content_id`、`content_version` | 内容标识和版本 |
| `family_id` | 内容族标识 |
| `kind` | 内容类型 |
| `assessment_id` | 关联测评标识，可为空 |
| `created_at` | 创建时间 |

### `content_review_events`：内容审核事件

记录题库题目版本的审核动作和审核备注。

| 字段 | 说明 |
| --- | --- |
| `id` | PostgreSQL identity 自增主键 |
| `item_id`、`item_version` | 被审核的题库项及版本 |
| `action` | 审核动作 |
| `review_note` | 审核备注 |
| `created_at` | 创建时间 |

### `courses`：课程

保存系统支持的课程。初始化时内置机器学习、高等数学、线性代数和概率论与统计基础。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，课程标识 |
| `title` | 课程名称 |
| `active` | 是否启用，`0` 或 `1` |

### `curriculum_topics`：课程主题

保存课程知识图谱中的主题、阶段顺序、学习目标和检索信息。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，主题标识 |
| `course` | 所属课程，必须是已注册课程 |
| `slug` | 主题短标识；与 `course` 联合唯一 |
| `title` | 主题名称 |
| `stage`、`stage_title` | 所属阶段编号和名称 |
| `order_index` | 阶段内排序 |
| `description` | 主题描述 |
| `learning_objectives_json` | 学习目标列表 |
| `aliases_json` | 主题别名列表 |
| `retrieval_keywords_json` | 检索关键词列表 |
| `difficulty` | 难度：`easy`、`medium`、`hard` |
| `required_dimensions_json` | 必须掌握的能力维度 |
| `active` | 是否启用 |
| `created_at`、`updated_at` | 创建和更新时间 |

### `documents`：知识库文档

保存上传文档的元数据、导入状态和切分结果。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，文档标识 |
| `course` | 所属课程 |
| `original_filename`、`stored_filename` | 原始文件名和存储文件名 |
| `sha256` | 文件内容哈希；与 `course` 联合唯一 |
| `file_size` | 文件大小 |
| `chunk_count` | 切分出的文本块数量 |
| `status` | 导入状态 |
| `error_message` | 导入失败信息，可为空 |
| `created_at`、`updated_at` | 创建和更新时间 |

### `document_chunks`：知识库文档片段与向量

保存文档切分后的可检索文本、来源元数据和 pgvector Embedding。每个片段关联一份知识库文档；删除文档时其片段会级联删除。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，文本片段标识 |
| `document_id` | 所属知识库文档，外键关联 `documents(id)` |
| `course` | 所属课程，用于课程隔离检索 |
| `content` | 切分后的文本内容 |
| `metadata` | JSONB 来源元数据，包括章节、页码、文件哈希和知识点标签 |
| `embedding` | pgvector 向量，用于余弦、L2 或内积相似度检索 |
| `created_at`、`updated_at` | 创建和更新时间 |

### `learning_sessions`：学习会话

记录学生进入某门课程或主题的学习会话。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，会话标识 |
| `student_id`、`course` | 学生和课程 |
| `topic_id` | 当前主题，可为空；主题删除时置空 |
| `state` | 会话状态 |
| `started_at`、`ended_at` | 开始和结束时间 |
| `created_at`、`updated_at` | 创建和更新时间 |

### `lessons`：课程正文

保存主题课程内容的版本化正文。

| 字段 | 说明 |
| --- | --- |
| `id`、`version` | 内容标识和版本，联合主键 |
| `topic_id` | 所属主题 |
| `status` | `draft`、`published` 或 `archived` |
| `payload_json` | 课程正文及其结构化内容 |
| `created_at` | 创建时间 |

### `misconception_events`：误解事件

记录学生在作答中暴露的误解，以及后续是否解决。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，事件标识 |
| `student_id`、`topic_id`、`attempt_id` | 学生、主题和对应作答 |
| `tag` | 误解标签 |
| `critical` | 是否为关键误解 |
| `evidence` | 误解证据 |
| `created_at`、`resolved_at` | 发现和解决时间 |

### `practice_attempts`：单题作答记录

保存一次提交中每道题的答案、评分、反馈和评价可靠性。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，作答记录标识 |
| `submission_id`、`question_id`、`student_id` | 所属提交、题目和学生 |
| `answer_text` | 学生答案 |
| `score`、`verdict` | 得分和评价结论 |
| `dimension_scores_json` | 各能力维度得分 |
| `correct_points_json`、`missing_points_json` | 已掌握和缺失要点 |
| `misconception_tags_json` | 误解标签 |
| `critical_misconception` | 是否存在关键误解 |
| `feedback`、`reference_answer` | 反馈和参考答案 |
| `evidence_source_ids_json` | 评价引用的知识来源 |
| `evaluator_confidence` | 评价器置信度 |
| `evaluation_reliable` | 评价是否可靠 |
| `created_at` | 创建时间 |
| 唯一约束 | `(submission_id, question_id)`，同一提交中一道题只能有一条记录 |

### `practice_questions`：练习题

保存练习集中的题目、答案规则、来源和作答辅助状态。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，题目标识 |
| `practice_set_id`、`position` | 所属练习集和题目序号；二者联合唯一 |
| `question_type`、`difficulty` | 题型和难度 |
| `question`、`options_json` | 题干和选项 |
| `dimensions_json`、`competencies_json` | 能力维度和能力项 |
| `answer_key_json`、`rubric_json` | 答案和评分规则 |
| `source_ids_json` | 知识来源标识 |
| `question_hash` | 题目内容哈希 |
| `created_at` | 创建时间 |
| `bank_item_id`、`item_version`、`family_id` | 题库项、版本和题族信息 |
| `topic_ids_json` | 关联主题列表 |
| `solution_steps_json` | 解题步骤 |
| `used_hint`、`solution_revealed` | 是否使用提示、是否展示解答 |
| `evidence_eligible` | 是否可作为掌握度证据 |
| `attempt_count` | 作答次数 |

### `practice_sets`：练习集

保存一次练习或检查点的组卷请求、来源快照和提交状态。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，练习集标识 |
| `student_id`、`topic_id` | 学生和目标主题 |
| `course` | 所属课程 |
| `mode` | `practice` 或 `checkpoint` |
| `status` | `generated`、`submitted` 或 `evaluation_uncertain` |
| `requested_difficulty` | `adaptive`、`easy`、`medium` 或 `hard` |
| `source_snapshot_json` | 生成时使用的知识来源快照 |
| `target_misconception_tags_json` | 目标误解标签 |
| `created_at`、`submitted_at` | 创建和提交时间 |
| `blueprint_id` | 使用的测评蓝图，可为空 |
| `purpose` | 练习目的，默认 `legacy` |
| `prerequisite_for` | 用于验收哪个主题，可为空 |
| `rule_version` | 组卷规则版本 |
| `required_dimensions_snapshot_json` | 生成时要求的能力维度快照 |

### `practice_submissions`：练习提交

保存学生提交整套练习后的汇总评分、评价状态和掌握度变化。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，提交标识 |
| `practice_set_id`、`student_id` | 所属练习集和学生 |
| `client_submission_id` | 客户端提交幂等标识，唯一 |
| `total_score` | 总分，范围 `0` 到 `100` |
| `evaluation_status` | `processing`、`evaluated` 或 `evaluation_uncertain` |
| `mastery_before`、`mastery_after` | 评价前后的掌握度 |
| `gate_result_json` | 前置门禁结果 |
| `response_json` | 原始或结构化响应 |
| `created_at` | 创建时间 |

### `prerequisite_verifications`：前置能力验证

保存学生通过练习集验证某个依赖主题能力的记录。

| 字段 | 说明 |
| --- | --- |
| `student_id`、`dependent_topic_id`、`prerequisite_topic_id` | 学生、依赖主题和前置主题 |
| `competency` | 已验证的能力项 |
| `rule_version` | 验证规则版本 |
| `assessment_id` | 用于验证的练习集 |
| `verified_at` | 验证时间 |
| 主键 | 学生、依赖主题、前置主题、能力项和规则版本的联合主键 |

### `question_bank_items`：题库题目

保存经过内容审核和版本管理的题库题目。

| 字段 | 说明 |
| --- | --- |
| `id`、`version` | 题库项标识和版本，联合主键 |
| `primary_course` | 主课程 |
| `question_type`、`difficulty` | 题型和难度 |
| `family_id` | 题族标识 |
| `publication_status` | 发布状态 |
| `validation_status` | 校验状态 |
| `payload_json` | 题目完整内容 |
| `created_at`、`updated_at` | 创建和更新时间 |

### `question_bank_topics`：题库题目主题关联

建立题库题目版本与课程主题之间的多对多关系。

| 字段 | 说明 |
| --- | --- |
| `item_id`、`item_version` | 题库项及版本，外键关联 `question_bank_items` |
| `topic_id` | 关联主题 |
| 主键 | `item_id`、`item_version`、`topic_id` 的联合主键 |

### `student_dimension_states`：学生能力维度状态

记录学生在主题下各项能力维度上的细粒度掌握情况。

| 字段 | 说明 |
| --- | --- |
| `student_id`、`topic_id`、`dimension` | 学生、主题和能力维度，联合主键 |
| `alpha`、`beta`、`mastery`、`confidence` | 概率模型参数、掌握度和置信度 |
| `evidence_count` | 证据数量 |
| `updated_at` | 更新时间 |

### `student_topic_states`：学生主题掌握状态

记录学生在每个主题上的掌握度、证据数量、练习次数和学习状态。

| 字段 | 说明 |
| --- | --- |
| `student_id`、`topic_id` | 学生和主题，联合主键 |
| `alpha`、`beta` | 掌握度概率模型参数 |
| `mastery`、`confidence` | 掌握度和置信度，范围均为 `0` 到 `1` |
| `evidence_count` | 有效证据数量 |
| `attempt_count` | 作答次数 |
| `practice_set_count`、`checkpoint_count` | 练习集和检查点数量 |
| `latest_score`、`best_score`、`latest_checkpoint_score` | 最近、最高和最近检查点评分 |
| `status` | `locked`、`not_started`、`learning`、`practicing`、`needs_remediation`、`ready_for_checkpoint` 或 `mastered` |
| `started_at`、`last_practiced_at`、`mastered_at`、`updated_at` | 学习生命周期时间 |

该表通过 `(student_id, topic_id)` 外键关联 `student_topic_states`，主题状态删除时级联删除。

### `students`：学生账户

保存学生账户、认证信息和活跃状态。匿名学习身份仅使用 `id` 和时间字段；注册账户同时保存用户名、显示名称和密码哈希。

| 字段 | 说明 |
| --- | --- |
| `id` | 主键，学生标识 |
| `username` | 登录用户名，可为空且全局唯一 |
| `display_name` | 页面展示名称，可为空 |
| `password_hash` | 密码哈希，可为空，不保存明文密码 |
| `active` | 账户是否启用，`0` 或 `1`，默认 `1` |
| `created_at` | 创建时间 |
| `last_active_at` | 最近活跃时间 |

### `topic_placements`：主题课程编排

保存主题在课程目录分组中的展示位置，支持同一主题作为参考内容出现在多个位置。

| 字段 | 说明 |
| --- | --- |
| `course_id`、`topic_id` | 课程和主题，联合主键 |
| `parent_group_id`、`group_title` | 所属目录分组及名称 |
| `order_index` | 分组内排序 |
| `is_reference` | 是否为参考位置 |

### `topic_prerequisites`：主题前置关系

描述主题之间的前置、推荐或关联关系，并记录前置掌握要求。

| 字段 | 说明 |
| --- | --- |
| `topic_id` | 当前主题，级联删除 |
| `prerequisite_topic_id` | 前置主题，禁止被引用时删除 |
| `required_mastery` | 要求的掌握度，范围 `0` 到 `1` |
| `relation_type` | `required`、`recommended` 或 `related` |
| `reason` | 建立该关系的原因 |
| `required_competencies_json` | 所需能力维度 |
| `scope`、`section_id` | 关系作用范围及章节区段 |
| `rule_version` | 规则版本 |
| 主键 | `topic_id`、`prerequisite_topic_id`、`relation_type`、`scope`、`section_id` 的联合主键 |

### `worked_examples`：例题

保存主题下可复用的版本化例题。

| 字段 | 说明 |
| --- | --- |
| `id`、`version` | 例题标识和版本，联合主键 |
| `topic_id` | 所属主题 |
| `family_id` | 例题族标识 |
| `status` | `draft`、`published` 或 `archived` |
| `payload_json` | 例题、解答和元数据 |
| `created_at` | 创建时间 |

![](.Figures/数据库表结构.png)

## 数据关系和完整性规则

- `students` 是学习过程的根实体；学生删除时，其主题状态、练习、提交、作答、误解事件和学习会话会按外键级联删除。
- `courses`、`curriculum_topics` 和 `topic_prerequisites` 构成课程目录与前置知识图谱。主题不能依赖自身；`required` 类型的前置关系不能形成环。
- `practice_sets` -> `practice_questions`、`practice_submissions` -> `practice_attempts` 形成练习、提交和单题评价链路。删除练习集会级联删除题目和提交，删除提交会级联删除作答记录。
- `lessons`、`worked_examples` 和 `question_bank_items` 使用 `(id, version)` 联合主键保存可追溯的内容版本；题库题目与主题通过 `question_bank_topics` 关联。
- 新增或更新 `curriculum_topics` 时，触发器会检查课程是否存在且处于启用状态；新增或更新必需前置关系时，触发器会阻止依赖环。

## 主要索引

初始化脚本为课程主题排序、前置主题查询、学生掌握状态、练习集与提交历史、误解事件、学习会话、题库筛选、内容曝光、文档课程/更新时间及文档片段的课程/文档关联建立索引，以支持课程导航、学习仪表盘、练习评价和知识库管理等查询。
