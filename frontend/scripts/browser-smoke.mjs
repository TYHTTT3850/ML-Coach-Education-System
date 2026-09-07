// Browser regression checks use only mocked HTTP responses; no real API or database writes.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir } from "node:fs/promises";
import process from "node:process";
import path from "node:path";

const { chromium } = createRequire(import.meta.url)(process.env.COURSE_PLAYWRIGHT_MODULE || "playwright");
const baseUrl = process.env.COURSE_UI_BASE_URL || "http://127.0.0.1:3030";
const browser = await chromium.launch({ headless: true, ...(process.env.COURSE_BROWSER_PATH ? { executablePath: process.env.COURSE_BROWSER_PATH } : {}) });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
const page = await context.newPage();
const problems = [];
const unhandled = [];
page.on("pageerror", (error) => problems.push(error.message));
const topics = [
  { id: "ml.linear_regression", course: "machine_learning", title: "线性回归", stage: 2, stage_title: "第二阶段：基础模型与训练", order_index: 1 },
  { id: "ml.gradient_descent", course: "machine_learning", title: "梯度下降", stage: 2, stage_title: "第二阶段：基础模型与训练", order_index: 2 },
  { id: "am.partial_derivatives", course: "advanced_mathematics", title: "偏导数与梯度", stage: 1, stage_title: "高等数学基础", order_index: 1 },
  { id: "ml.machine_learning_basics", course: "machine_learning", title: "机器学习基础", stage: 1, stage_title: "第一阶段：机器学习入门", order_index: 1 },
];
const topicData = (topic) => ({ ...topic, owner_course: topic.course, topic_id: topic.id, description: "用两条样本计算预测值、损失和一次参数更新。", learning_objectives: ["能够计算线性模型的一次预测与损失。"], aliases: [], difficulty: "easy", required_dimensions: ["calculation"], prerequisites: [], dimensions: [], status: "learning", mastery: null, confidence: null, evidence_count: 0, prerequisite_gap: [], content_status: "ready", can_checkpoint: true, is_reference: topic.course !== "machine_learning" });
let verified = false;
let prerequisiteFailure = "temporary";
let studentCreations = 0;
let serial = 0;
const instances = new Map();
const instanceEngines = new Map();
const requests = [];
let generationFailure = false;
const hiddenMarker = "CHECKPOINT_SECRET_NOT_BEFORE_SUBMIT";
const forbidden = new Set(["answer_key", "rubric", "solution_steps", "source_snapshot", "accepted_variants"]);
function assertPublic(value) {
  if (Array.isArray(value)) { value.forEach(assertPublic); return; }
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) { assert(!forbidden.has(key), `Hidden field in public assessment: ${key}`); assertPublic(child); }
}
const edge = () => ({ id: "required-grad", prerequisite_topic_id: topics[2].id, dependent_topic_id: topics[0].id, title: topics[2].title, course: topics[2].course, relation_type: "required", reason: "多参数更新需要由偏导数组成梯度。", required_competencies: ["计算偏导数组成的梯度"], scope: "topic", satisfied: verified, verified_competencies: verified ? ["计算偏导数组成的梯度"] : [] });
const example = { id: "example-linear", topic_id: topics[0].id, title: "两条样本的一次更新", difficulty: "easy", problem: "给定两条样本，计算一次参数更新。", givens: { x: [1, 2], y: [2, 4], w: 1, b: 0 }, target: "求更新后的参数和损失", solution_outline: "预测 → 损失 → 偏导数 → 更新。", steps: [{ step_id: "predict", title: "第一步：计算预测", explanation: "把每条样本代入模型。", formula: "\\hat y=wx+b", substitution: "第一条：1×1+0=1；第二条：1×2+0=2。", result: "预测为 [1, 2]。", justification: "逐条代入线性模型" }], final_answer: "例题演示结果", verification: {}, common_mistakes: ["不要把梯度当作损失值。"], followup_practice_ids: [], source_refs: [], content_origin: "instructor_authored", version: 1, status: "published", family_id: "example-only-family" };
const lesson = { id: "lesson-linear", topic_id: topics[0].id, title: "线性回归", learning_objectives: ["计算两条样本的损失"], prerequisite_summary: "向量和求导基础", motivating_problem: "房屋面积为 1、2 时，价格为 2、4，怎样调整直线？", key_concepts: ["用预测与真实值的差计算误差。"], formulas: [{ expression: "L=\\frac{1}{2n}\\sum_i(\\hat y_i-y_i)^2", symbols: "n 是样本数量；预测值和标签均为标量。", conditions: "n > 0，采用平方损失。" }], worked_example_ids: [example.id], practice_recommendations: ["bank.ml.linear_regression.practice.1", "独立计算另一组预测误差。"], summary: "参数更新后需要重新检查损失。", source_refs: [], content_origin: "instructor_authored", version: 1, status: "published" };
const groups = [{ id: "math-preparation", title: "数学准备", parent_group_id: null, order_index: 0, topics: [] }, { id: "math-preparation-advanced_mathematics", title: "高等数学", parent_group_id: "math-preparation", order_index: 0, topics: [topicData(topics[2])] }, ...Array.from({ length: 7 }, (_, index) => ({ id: `stage-${index + 1}`, title: `第${index + 1}阶段`, parent_group_id: null, order_index: index + 1, topics: index === 1 ? topics.slice(0, 2).map(topicData) : [] }))];
example.source_refs.push({ source_id: "authored:browser-example", document_id: "browser-fixture", source_file: "测试自编例题", chunk_index: 0, snippet: "浏览器回归中的已保存示例说明。" });
const json = (route, value, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(value), headers: { "Access-Control-Allow-Origin": "*" } });
await context.route(/\/api\/v1\//, async (route) => {
  const request = route.request(); const url = new URL(request.url()); const endpoint = url.pathname; const body = request.postData() ? request.postDataJSON() : {};
  if (request.method() !== "OPTIONS") requests.push({ endpoint, body, method: request.method() });
  if (request.method() === "OPTIONS") return route.fulfill({ status: 204, headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Methods": "*" } });
  if (endpoint === "/api/v1/students/anonymous") return json(route, { student_id: `browser-mock-student-${++studentCreations}`, created_at: "2026-09-06T00:00:00Z" });
  if (endpoint.includes("/courses/") && endpoint.endsWith("/outline")) return json(route, { course: "machine_learning", title: "机器学习", groups, recommended_topic_id: topics[0].id });
  if (endpoint === "/api/v1/curriculum/topics") return json(route, { course: url.searchParams.get("course"), stages: [{ stage: 2, title: "基础模型", topics: topics.filter((topic) => topic.course === url.searchParams.get("course")).map(topicData) }] });
  if (endpoint.startsWith("/api/v1/curriculum/topics/")) return json(route, topicData(topics.find((topic) => endpoint.endsWith(topic.id)) || topics[0]));
  if (endpoint.endsWith("/prerequisites")) {
    if (prerequisiteFailure === "temporary") return json(route, { error: { code: "PREREQUISITE_TEMP_UNAVAILABLE", message: "前置服务暂时无法连接。" } }, 503);
    if (prerequisiteFailure === "stale") {
      prerequisiteFailure = "";
      return json(route, { error: { code: "STUDENT_NOT_FOUND", message: "匿名学生不存在。" } }, 404);
    }
    return json(route, { topic_id: topics[0].id, owner_course: "machine_learning", rule_version: "content-v1", can_browse: true, can_study: true, can_checkpoint: verified, required: [edge()], recommended: [], related: [], unmet: verified ? [] : [edge()], successors: [] });
  }
  if (endpoint.endsWith("/lesson")) return endpoint.includes(topics[3].id) ? json(route, { detail: "Fixed lesson not published" }, 404) : json(route, lesson);
  if (endpoint.endsWith("/examples")) return json(route, { examples: endpoint.includes(topics[3].id) ? [] : [example] });
  if (endpoint.endsWith("/exposure")) return json(route, { recorded: true });
  if (endpoint.endsWith("/ask")) return json(route, { answer: "步骤答疑只使用版本 1 的真实例题。", sources: [] });
  if (endpoint === "/api/v1/chat") {
    assert.equal(body.learning_mode, "worked_example");
    assert.equal(body.course, "machine_learning");
    assert(body.topic_id && body.message.includes("2 道"));
    return json(route, { answer: "教学例题一：先辨认输入与目标。\n\n教学例题二：代入具体数据并检查结论。", sources: [{ source_id: "mock-course-source", document_id: "browser-fixture", source_file: "课程资料测试样本", chunk_index: 0, snippet: "例题基于检索的知识点课程资料。" }] });
  }
  if (endpoint === "/api/v1/learning/history") return json(route, { course: "machine_learning", events: [] });
  if (endpoint.endsWith("/start")) return json(route, { topic: topicData(topics[0]) });
  if (endpoint === "/api/v1/learning/dashboard") return json(route, { course: "machine_learning", total_topics: 2, mastered_count: 0, learning_count: 1, not_started_count: 1, locked_count: 0, current_topic: topicData(topics[0]), recommended_topic_id: topics[0].id, next_action: "teach", reasons: [], recent_practice: [], stages: [] });
  if (endpoint === "/api/v1/content/coverage") return json(route, { topics: topics.map((topic) => ({ topic_id: topic.id, title: topic.title, owner_course: topic.course, lesson_count: 1, example_count: 2, practice_count: 4, checkpoint_count: 5, published_count: 12, needs_review_count: 0, prerequisites_ready: true, full_flow_ready: true })) });
  if (endpoint === "/api/v1/question-bank") return json(route, { items: [{ id: "bank-practice", version: 1, topic_ids: [topics[0].id], primary_course: "machine_learning", title: "独立练习", question_type: "numeric", difficulty: "easy", stem: "计算另一个模型的预测。", options: null, tested_competencies: ["calculation"], estimated_minutes: 2, usage: ["practice"], content_origin: "instructor_authored", publication_status: "published", validation_status: "deterministic_checks_passed" }], checkpoint_count: 5 });
  if (endpoint === "/api/v1/assessments" || endpoint === "/api/v1/prerequisite-tests" || endpoint === "/api/v1/practice/sets") {
    const dynamic = endpoint === "/api/v1/practice/sets";
    if (dynamic) {
      assert.equal(body.count, body.mode === "checkpoint" ? 5 : 3);
      assert.equal(body.difficulty, "adaptive");
      assert.equal(body.topic_id, topics[0].id);
      if (generationFailure) return json(route, { error: { code: "KNOWLEDGE_INSUFFICIENT", message: "当前知识点课程资料不足，请补充资料后重试。" } }, 422);
    }
    if (endpoint === "/api/v1/assessments" && body.purpose === "practice" && !body.bank_item_ids?.length) {
      assert.equal(body.exclude_seen, true, "Default practice must use unseen evidence");
      assert.equal(body.exclude_exposed_families, true, "Default practice must exclude exposed example families");
    }
    const purpose = endpoint.endsWith("prerequisite-tests") ? "prerequisite_test" : dynamic ? body.mode : body.purpose;
    const count = dynamic ? body.count : body.question_count;
    const instance = { practice_set_id: `mock-assessment-${++serial}`, topic_id: body.topic_ids?.[0] || topics[0].id, purpose, mode: purpose === "practice" ? "practice" : "checkpoint", questions: Array.from({ length: count }, (_, index) => ({ question_id: `mock-${serial}-${index}`, position: index + 1, question_type: dynamic ? ["numeric", "concept_explanation", "code_reading"][index % 3] : "numeric", difficulty: "easy", question: `课程资料题 ${index + 1}：计算一个数值或说明理由。`, dimension_names: ["calculation"], required: true })), source_summary: [{ source_id: "authored:bank.ml.linear_regression.independent.practice.long-family-name:v1", source_file: dynamic ? "课程资料测试样本" : "项目自编教学示例", title: "独立计算练习" }], result: null };
    instances.set(instance.practice_set_id, instance); instanceEngines.set(instance.practice_set_id, dynamic ? "dynamic" : "assessment"); assertPublic(instance); return json(route, instance);
  }
  const assessmentMatch = endpoint.match(/\/(?:assessments|practice\/sets)\/([^/]+)(?:\/(submit|hint|solution))?$/);
  if (assessmentMatch) {
    const instance = instances.get(assessmentMatch[1]); assert(instance, "Unknown mocked assessment");
    if (assessmentMatch[2] === "hint") return json(route, { hint: "先写清参数和输入。" });
    if (assessmentMatch[2] === "solution") return json(route, { answer: "练习参考值", solution_steps: ["先代入，再求和。"] });
    if (assessmentMatch[2] === "submit") {
      assert.equal(endpoint.includes("/practice/sets/"), instanceEngines.get(instance.practice_set_id) === "dynamic", "Submission endpoint must follow the saved instance, not the page default");
      if (instance.purpose === "prerequisite_test") verified = true;
      instance.result = { practice_set_id: instance.practice_set_id, total_score: 100, mastery_before: null, mastery_after: 0.6, evaluations: instance.questions.map((question) => ({ question_id: question.question_id, score: 100, verdict: "正确", feedback: "计算步骤正确。", correct_points: ["计算正确"], missing_points: [], misconception_tags: [], reference_answer: hiddenMarker, sources: [], evaluation_reliable: true })), gate: { passed: false, next_action: "practice", requirements: {} }, status: "completed" };
      return json(route, instance.result);
    }
    if (!instance.result) assertPublic(instance);
    return json(route, { ...instance, purpose: instanceEngines.get(instance.practice_set_id) === "dynamic" ? "legacy" : instance.purpose });
  }
  unhandled.push(`${request.method()} ${endpoint}`);
  return json(route, { detail: "Unexpected mocked endpoint" }, 500);
});

async function check(name, run) { await run(); process.stdout.write(`PASS ${name}\n`); }
async function fillAll(value = "2") { for (const input of await page.locator('#practice input[inputmode="decimal"], #practice textarea').all()) await input.fill(value); }
try {
  await page.goto(`${baseUrl}/learn/machine_learning/ml.linear_regression`);
  await page.getByRole("heading", { name: "线性回归", exact: true }).waitFor();
  await check("directory, search, collapse and current topic", async () => {
    await page.getByText(/前置能力状态尚未确认/).waitFor();
    await page.getByRole("alert").filter({ hasText: "PREREQUISITE_TEMP_UNAVAILABLE" }).waitFor();
    assert.equal(await page.getByText(/前置已满足/).count(), 0, "Unknown prerequisite state must not be presented as satisfied");
    assert(await page.getByRole("button", { name: "AI 生成 5 道综合验收", exact: true }).isDisabled());
    assert(await page.getByRole("button", { name: "AI 生成 3 道练习", exact: true }).isEnabled());
    prerequisiteFailure = "stale";
    await page.getByRole("button", { name: "重试", exact: true }).click();
    await page.locator("#prerequisites").waitFor();
    assert.equal(studentCreations, 2, "A stale identity inside a partial-result batch must recover exactly once");
    const sidebar = page.getByRole("complementary", { name: "课程知识点目录" });
    await sidebar.getByText("数学准备", { exact: true }).waitFor();
    assert.equal(await sidebar.locator('[aria-current="page"]').count(), 1);
    await page.getByLabel("搜索知识点").fill("梯度下降");
    assert.equal(await sidebar.getByRole("link", { name: /线性回归/ }).count(), 0);
    await page.getByLabel("搜索知识点").fill("");
    await page.getByRole("button", { name: "折叠课程目录", exact: true }).click();
    await page.getByRole("button", { name: "展开课程目录", exact: true }).click();
    await sidebar.getByRole("button", { name: /第2阶段/ }).click();
    await page.getByRole("link", { name: /下一个：梯度下降/ }).click();
    await page.getByRole("heading", { name: "梯度下降", exact: true }).waitFor();
    await sidebar.locator('[aria-current="page"]').waitFor();
    await page.goBack();
    await page.getByRole("heading", { name: "线性回归", exact: true }).waitFor();
  });
  await check("canonical math prerequisite and return context", async () => {
    await page.locator("#prerequisites").getByRole("link", { name: "偏导数与梯度 ↗" }).click();
    await page.getByRole("heading", { name: "偏导数与梯度", exact: true }).waitFor();
    assert(page.url().includes("/advanced_mathematics/am.partial_derivatives?course=machine_learning"));
    await page.getByRole("link", { name: /返回原机器学习知识点/ }).click();
    await page.getByRole("heading", { name: "线性回归", exact: true }).waitFor();
    await page.reload(); await page.getByRole("heading", { name: "线性回归", exact: true }).waitFor();
  });
  await check("concrete lesson, worked steps and step question", async () => {
    await page.getByText("从这个问题开始", { exact: true }).waitFor();
    await page.getByRole("link", { name: "练习 1：检验本节学习目标", exact: true }).waitFor();
    assert(!(await page.locator("#lesson").innerText()).includes("bank.ml."), "Practice recommendation IDs must not appear as student text");
    await page.getByText("例题来源", { exact: true }).click();
    await page.getByText("浏览器回归中的已保存示例说明。", { exact: true }).waitFor();
    await page.getByText("展开完整分步解法", { exact: true }).click();
    await page.getByText("第一步：计算预测", { exact: true }).waitFor();
    await page.getByText("针对这一步提问", { exact: true }).click();
    await page.getByLabel("你的疑问", { exact: true }).fill("为什么要逐条代入？");
    await page.getByRole("button", { name: "询问这个步骤", exact: true }).click();
    await page.getByText("步骤答疑只使用版本 1 的真实例题。", { exact: true }).waitFor();
    assert.equal(requests.filter((item) => item.endpoint === "/api/v1/chat").length, 0, "Opening a lesson must not auto-call AI");
    await page.getByRole("button", { name: "AI 讲例题", exact: true }).click();
    await page.getByText("教学例题一：先辨认输入与目标。", { exact: true }).waitFor();
    await page.getByText("查看 1 条课程资料来源", { exact: true }).click();
    await page.getByText("例题基于检索的知识点课程资料。", { exact: true }).waitFor();
  });
  await check("prerequisite quiz opens formal checkpoint locally", async () => {
    assert(await page.getByRole("button", { name: "AI 生成 5 道综合验收", exact: true }).isDisabled());
    await page.getByRole("button", { name: "做前置小测", exact: true }).click();
    await page.getByRole("heading", { name: "前置能力小测", exact: true }).waitFor();
    await fillAll(); await page.getByRole("button", { name: "提交整套答案", exact: true }).click();
    await page.getByRole("region", { name: "学习结果" }).waitFor();
    await page.getByRole("button", { name: "AI 生成 5 道综合验收", exact: true }).waitFor();
    assert(await page.getByRole("button", { name: "AI 生成 5 道综合验收", exact: true }).isEnabled());
  });
  await check("AI dynamic practice, varied answer forms and refresh recovery", async () => {
    const bankBefore = requests.filter((item) => item.endpoint === "/api/v1/assessments").length;
    generationFailure = true;
    await page.getByRole("button", { name: "AI 生成 3 道练习", exact: true }).click();
    await page.getByText("当前知识点课程资料不足，请补充资料后重试。", { exact: true }).waitFor();
    assert.equal(requests.filter((item) => item.endpoint === "/api/v1/assessments").length, bankBefore, "Insufficient knowledge must not silently switch to bank questions");
    generationFailure = false;
    await page.getByRole("button", { name: "AI 生成 3 道练习", exact: true }).click();
    await page.getByRole("heading", { name: "主动练习", exact: true }).waitFor();
    assert.equal(await page.locator('#practice input[inputmode="decimal"]').count(), 1);
    assert.equal(await page.locator("#practice textarea").count(), 2);
    await fillAll("7");
    const created = serial;
    await page.reload(); await page.getByRole("heading", { name: "主动练习", exact: true }).waitFor();
    assert.equal(serial, created, "Refreshing must not reassemble questions");
    assert.equal(await page.locator('input[inputmode="decimal"]').first().inputValue(), "7");
    assert.equal(await page.locator("#practice textarea").first().inputValue(), "7");
    assert(await page.getByRole("button", { name: "AI 讲例题", exact: true }).isEnabled(), "Restored legacy practice is not a formal checkpoint");
    await page.getByRole("button", { name: "提交整套答案", exact: true }).click();
    await page.getByRole("region", { name: "学习结果" }).waitFor();
    await page.reload(); await page.getByRole("region", { name: "学习结果" }).waitFor();
  });
  await check("checkpoint keeps hidden material out of HTML and responses", async () => {
    await page.getByRole("button", { name: "AI 生成 5 道综合验收", exact: true }).click();
    await page.getByRole("heading", { name: "知识点综合验收", exact: true }).waitFor();
    assert.equal(await page.getByRole("button", { name: "请求提示", exact: true }).count(), 0);
    assert.equal(await page.getByRole("button", { name: "AI 讲例题", exact: true }).count(), 0);
    assert(!(await page.content()).includes(hiddenMarker));
    const created = serial; await page.reload();
    await page.getByRole("heading", { name: "知识点综合验收", exact: true }).waitFor();
    assert.equal(serial, created); assert(!(await page.content()).includes(hiddenMarker));
    await fillAll(); await page.getByRole("button", { name: "提交整套答案", exact: true }).click();
    await page.getByRole("region", { name: "学习结果" }).waitFor();
    assert((await page.content()).includes(hiddenMarker), "References become available only after submit");
  });
  await check("mobile drawer, keyboard focus and formula width", async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("button", { name: "☰ 课程目录", exact: true }).click();
    await page.getByRole("button", { name: "关闭课程目录", exact: true }).waitFor();
    await page.keyboard.press("Escape");
    assert.equal(await page.getByRole("button", { name: "☰ 课程目录", exact: true }).getAttribute("aria-expanded"), "false");
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), "Page must not overflow horizontally");
  });
  await check("student question bank and content coverage", async () => {
    await page.goto(`${baseUrl}/question-bank`);
    await page.getByRole("heading", { name: "独立练习", exact: true }).waitFor();
    await page.getByRole("heading", { name: "内容覆盖报告", exact: true }).waitFor();
    assert(!(await page.content()).includes(hiddenMarker));
    await page.getByRole("button", { name: "练习这道题", exact: true }).click();
    await page.getByRole("heading", { name: "主动练习", exact: true }).waitFor();
    assert.equal(await page.locator('input[inputmode="decimal"]').count(), 1);
    await fillAll("7");
    await page.getByRole("button", { name: "请求提示", exact: true }).first().click();
    await page.getByText("先写清参数和输入。", { exact: true }).waitFor();
    await page.getByRole("button", { name: "已尝试，查看解法", exact: true }).first().click();
    await page.getByText("参考答案：练习参考值", { exact: true }).waitFor();
    const created = serial;
    await page.reload(); await page.getByRole("heading", { name: "主动练习", exact: true }).waitFor();
    assert.equal(serial, created);
    assert.equal(await page.locator('input[inputmode="decimal"]').first().inputValue(), "7");
  });
  await check("old bank instance without engine survives new dynamic course default", async () => {
    await page.evaluate(() => {
      for (const key of Object.keys(localStorage)) {
        if (key.startsWith("course-coach:assessment:")) {
          const value = JSON.parse(localStorage.getItem(key)); delete value.engine;
          localStorage.setItem(key, JSON.stringify(value));
        }
      }
    });
    await page.goto(`${baseUrl}/learn/machine_learning/ml.linear_regression`);
    await page.getByRole("heading", { name: "主动练习", exact: true }).waitFor();
    await page.getByRole("button", { name: "请求提示", exact: true }).first().waitFor();
    assert.equal(await page.locator('input[inputmode="decimal"]').count(), 1);
    await page.getByRole("button", { name: "提交整套答案", exact: true }).click();
    await page.getByRole("region", { name: "学习结果" }).waitFor();
  });
  await check("AI examples remain available without a fixed lesson and reset by topic", async () => {
    await page.goto(`${baseUrl}/learn/machine_learning/ml.machine_learning_basics`);
    await page.getByRole("heading", { name: "机器学习基础", exact: true }).waitFor();
    await page.getByRole("heading", { name: "□ 固定课程正文待补充", exact: true }).waitFor();
    assert.equal(await page.getByText("教学例题一：先辨认输入与目标。", { exact: true }).count(), 0);
    const before = requests.filter((item) => item.endpoint === "/api/v1/chat").length;
    await page.getByRole("button", { name: "AI 讲例题", exact: true }).click();
    await page.getByText("教学例题一：先辨认输入与目标。", { exact: true }).waitFor();
    const chats = requests.filter((item) => item.endpoint === "/api/v1/chat");
    assert.equal(chats.length, before + 1);
    assert.equal(chats.at(-1).body.topic_id, topics[3].id);
  });
  const artifactDirectory = path.resolve(".browser-artifacts"); await mkdir(artifactDirectory, { recursive: true });
  await page.screenshot({ path: path.join(artifactDirectory, "dynamic-examples-mobile.png"), fullPage: true });
  assert.deepEqual(problems, [], "Browser runtime errors"); assert.deepEqual(unhandled, [], "Unhandled API mocks");
  process.stdout.write("10 browser scenarios passed; all API calls mocked.\n");
} finally { await browser.close(); }
