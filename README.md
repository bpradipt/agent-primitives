# Claude Code Commands and Skills

A collection of [Claude Code](https://claude.ai/code) tools for specialised development workflows. This repo contains two types of artefacts: **slash commands** and **skills**.

---

## Types

### Slash commands

Installed in `~/.claude/commands/`. Invoked explicitly by the user with `/command-name [args]`. The `.md` file includes `argument-hint` and `allowed-tools` frontmatter fields.

| Name | Invocation | Description |
|---|---|---|
| [cloud-api-adaptor](./cloud-api-adaptor/) | `/cloud-api-adaptor [setup\|build\|test]` | Set up, build, and test cloud-api-adaptor with libvirt locally (amd64, Ubuntu 24.04 podvm via mkosi) |

### Skills

Installed in `~/.claude/skills/`. Claude invokes them automatically via the `Skill` tool when the task matches the skill's description. The skill file only needs `name` and `description` frontmatter.

| Name | Trigger | Description |
|---|---|---|
| [mlx-vlm-multispectral-finetune](./mlx-vlm-multispectral-finetune/) | Auto (by Claude) | Fine-tune a vision-language model on multi-spectral satellite imagery (Sentinel-2, Landsat) using mlx-vlm on Apple Silicon with LoRA |

---

## Installation

### Prerequisites

- [Claude Code](https://claude.ai/code) CLI installed

### Install a slash command

```bash
git clone https://github.com/bpradipt/agent-skills
cp agent-skills/<skill-name>/<skill-name>.md ~/.claude/commands/
```

### Install a skill

```bash
git clone https://github.com/bpradipt/agent-skills
cp agent-skills/<skill-name>/SKILL.md ~/.claude/skills/<skill-name>.md
```

### Install all slash commands

```bash
git clone https://github.com/bpradipt/agent-skills
for skill_dir in agent-skills/*/; do
    skill_name=$(basename "$skill_dir")
    md="$skill_dir/$skill_name.md"
    [ -f "$md" ] && cp "$md" ~/.claude/commands/
done
```

Claude Code picks up files in `~/.claude/commands/` and `~/.claude/skills/` automatically — no restart required.

### Keep up to date

```bash
cd agent-skills
git pull
# Re-run the copy commands above
```

---

## Usage

### Slash commands

Invoke with `/command-name` in any Claude Code session:

```
/cloud-api-adaptor setup
/cloud-api-adaptor build --debug
/cloud-api-adaptor test --filter TestLibvirtCreatePeerPod
```

Run any slash command with no arguments for an interactive prompt.

### Skills

Claude detects when a skill applies and invokes it automatically. You can also trigger one explicitly via the `Skill` tool in Claude Code.

---

## Contributing

### Adding a slash command

```
<command-name>/
├── <command-name>.md   # YAML frontmatter (name, description, argument-hint, allowed-tools) + Markdown body
└── README.md           # Human-readable docs
```

Required frontmatter fields: `name`, `description`, `argument-hint`, `allowed-tools`. Use an existing slash command as a template.

### Adding a skill

```
<skill-name>/
└── SKILL.md            # YAML frontmatter (name, description) + Markdown body
```

Required frontmatter fields: `name`, `description`. The `description` is what Claude uses to decide when to invoke the skill — make it precise.

Open a pull request once your command or skill is ready.
