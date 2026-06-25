# Homework — Build the Working Surface for Your App

*This homework turns Part I into a concrete setup you can keep using for the rest of the course. The point is not to build a large application. The point is to create a small real project with enough structure that an AI coding agent can onboard, understand the domain, run the app, and make future changes with less guessing.*

## Topic

**Build a tiny app and the AI operating surface around it.**

## Main Goal

Create a deliberately small web app and package it with the context, rules, tools, and reference material an agent will need later. By the end, you should have a repo that is easy for both a human and an agent to navigate: the app runs, the intent is written down, the boundaries are explicit, and the useful commands and docs are discoverable without spelunking.

Keep the app itself narrow. A todo list, bookmarks catalogue, reading list, recipe box, habit tracker, or lightweight personal CRM are all perfectly good choices. In Week 1, you only need **one thin vertical slice**: one page or screen, one domain object, and one simple create-and-display flow. Minimal persistence is fine.

## Core Exercise

Build the app and add the core pieces of a portable context stack around it.

Your repository should include:

- a **working mini-app** with one clear user flow;
- `docs/spec.md` with the app’s **intent**, **acceptance criteria**, **constraints**, and **non-goals**;
- `AGENTS.md` or the equivalent rules file for your main tool, including the real run/build/test commands, key paths, durable docs, and clear **always / ask first / never** guidance;
- at least **two Skills**:
  - one **procedural Skill** such as `app-smoke-test`, `setup-and-run`, or `frontend-change-review`;
  - one **knowledge Skill** such as `app-domain-reference` or `data-model-and-route-map`;
- at least **one MCP server**, with Playwright MCP as the default recommendation for a web app;
- `docs/reference-index.md` or an equivalent curated doc index pointing to the framework docs, testing or browser-tool docs, and one additional source relevant to your stack;
- a `README.md` that explains what the app is, how to run it, which tool you used, and where the rules, Skills, MCP config, and reference docs live.

Work pragmatically. Start from a familiar starter if that saves time. Adapt existing examples where sensible. Keep the rules file short and operational rather than philosophical. Make the procedural Skill concrete enough that you would genuinely want to invoke it again, and make the knowledge Skill a tight discovery surface rather than a dumping ground for every note you have ever had.

The real learning target is the **division of labour between context surfaces**: rules for always-on defaults, Skills for specialised procedures or reference knowledge, MCP for runtime inspection, and docs for durable truth. If that division is still fuzzy in your repo, tighten it until it becomes obvious.

## Stretch Goal

Prove that the setup works rather than stopping at “the files exist”.

Add `docs/validation.md` and run a few fresh-session checks. Ask the agent which package manager and dev command to use, where the durable docs live, which Skill should handle a smoke test, which MCP server is available, and what is out of bounds. Then run one real workflow with the procedural Skill and the MCP server: launch the app, perform a simple user action, confirm the result, and ask the knowledge Skill to explain the domain model or suggest one small next change. If the answers wobble or turn generic, revise the setup once and record what improved.

## Advanced Stretch

If you want to push this further, make the setup more **shareable, portable, or harder to break**. Good directions include adding another procedural Skill, generating a lightweight code map, introducing a second MCP server that genuinely matches your workflow, or turning one of the Skills into a more deterministic script-backed procedure. You could also add a project-local AI-facing doc index, a second rules surface for another client, a tiny `docs/decisions.md`, or a minimal test harness that prepares the repo for Week 2.

The advanced bar is not about volume. One excellent reusable review Skill and one crisp architecture summary are worth far more than a forest of context files nobody will actually maintain.

