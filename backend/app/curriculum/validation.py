from __future__ import annotations

from collections.abc import Sequence

from app.curriculum.models import CurriculumSeed


class CurriculumValidationError(ValueError):
    pass


class CrossCourseDependencyError(CurriculumValidationError):
    pass


class CurriculumCycleError(CurriculumValidationError):
    def __init__(self, course: str, cycle: list[str]) -> None:
        self.course = course
        self.cycle = cycle
        super().__init__(f"课程 {course} 存在循环依赖: {' -> '.join(cycle)}")


def validate_curriculum(seeds: Sequence[CurriculumSeed]) -> None:
    """Validate canonical references and the global required DAG, across courses."""

    topic_courses: dict[str, str] = {}
    for seed in seeds:
        for topic in seed.topics:
            if topic.id in topic_courses:
                raise CurriculumValidationError(f"知识点 ID {topic.id} 在多个课程中重复")
            topic_courses[topic.id] = seed.course.value

    graph = {topic_id: [] for topic_id in topic_courses}
    for seed in seeds:
        for topic in seed.topics:
            for prerequisite in topic.prerequisites:
                dependency_course = topic_courses.get(prerequisite.topic_id)
                if dependency_course is None:
                    raise CurriculumValidationError(
                        f"知识点 {topic.id} 引用了不存在的前置知识点 {prerequisite.topic_id}"
                    )
                # Directed edge prerequisite -> topic.
                if prerequisite.relation_type == "required":
                    graph[prerequisite.topic_id].append(topic.id)
    _raise_for_cycle("global", graph)


def _raise_for_cycle(course: str, graph: dict[str, list[str]]) -> None:
    state: dict[str, int] = {node: 0 for node in graph}
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for neighbor in graph[node]:
            if state[neighbor] == 0:
                visit(neighbor)
            elif state[neighbor] == 1:
                start = stack.index(neighbor)
                raise CurriculumCycleError(course, stack[start:] + [neighbor])
        stack.pop()
        state[node] = 2

    for node in graph:
        if state[node] == 0:
            visit(node)
