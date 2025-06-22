# Towards Agentic LLM frameworks for automating GUI-Tasks: Comparing a single LLM to an Agentic Architecture

This repository contains the source code and experimental results for the bachelor thesis titled "Towards Agentic LLM frameworks for automating GUI-Tasks: Comparing a single LLM to an Agentic Architecture" by Frederik Bauer, submitted to the Vienna University of Economics and Business.

## 📖 Abstract

This thesis investigates whether a hierarchical agent architecture can improve Graphical User Interface (GUI) automation. We compared a manager-worker system, using GPT-4o for high-level planning and the UI-TARS model for low-level execution, against a baseline where UI-TARS operated as a single end-to-end agent. Both systems were evaluated on a set of tasks from the OS-World benchmark, measuring Task Success Rate (TSR) and analyzing failure modes.

The hierarchical system achieved a modestly higher overall TSR (23.3% vs. 16.7%), demonstrating a clear advantage on simple, structured tasks. However, this advantage disappeared with increasing complexity; both systems failed all 'Medium' difficulty web-based tasks, and the monolithic baseline performed better on 'Hard' tasks. Qualitative analysis revealed that even with correct high-level plans, the worker agent consistently failed at fundamental UI interactions like handling web forms and pop-ups.

We conclude that while architectural improvements in planning offer some benefits, they are ultimately undermined by persistent failures in low-level action execution. Robust GUI automation requires foundational improvements in the agent’s core interaction capabilities, highlighting that reliable execution is a prerequisite for high-level strategies to be effective.

## 🤖 System Architectures

Two primary architectures were implemented and compared in this study.

### 1. Hierarchical Manager-Worker System

This system distributes the cognitive load between a high-level planner and a low-level executor.

* **Manager (Planner):** `GPT-4o` is used to decompose a high-level user goal into a sequence of simpler, actionable steps.
* **Worker (Executor):** `UI-TARS-7B-DPO` receives one low-level instruction at a time from the manager and is responsible for the direct interaction with the GUI (e.g., clicks, typing).

*Figure 1: Hierarchical Manager-Worker Architecture*
![Hierarchical Manager-Worker Architecture](agenticLlmFrameworksForAutomatingGuiTasks/resources/ExecutionFlowAgenticArchitecture.png)
