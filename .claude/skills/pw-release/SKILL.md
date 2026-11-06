---
name: pw-release
description: Prepare a PluggyWorld release. Gathers everything on staging since the last release in pluggybot and rooftop-media-2026, drafts the release notes for people and the changelog entry the robots read (src/pluggybot/mind/changelog.md, under its rules), commits the entry to staging, opens both repos' staging-to-main "Release" PRs (the site's first), and stops for Ben's review. Never merges or deploys. Use when Ben asks to release, cut a release, prepare the release PRs, or write the changelog entry.
---

# /pw-release: the release PRs and the robots' changelog entry

A release is two PRs, `staging` into `main`, both titled **Release**, the
site's first (rooftop-media-2026), then the sim's (pluggybot). A push to
`main` IS the deploy in both repos (GitHub Actions on the box), so merging is
Ben's and only Ben's. This skill does everything up to that and stops.

It writes two texts:

1. **The release notes, for people**: each PR's body. What is in it, what
   needs Ben, the deploy order, what to read on the box after.
2. **The changelog entry, for the robots**: the dated entry at the top of
   `src/pluggybot/mind/changelog.md`. Luca and Rowan read it through
   `recall`, every turn carries its date, and the site shows it. It is a
   human's text: Ben approves every entry before it ships, in the pluggybot
   PR. The rules are docs/Overseer.md §7b and `tests/test_changelog.py`
   enforces them.

## ⚠ What this skill never does

- **Never merges, never deploys**, never pushes to `main`, never force-pushes
  anything. The one push it makes is the changelog commit onto `staging`,
  fast-forward only.
- **Never copies a commit message or a PR body into the changelog.** They
  carry what we measured, why, the hypotheses and the hidden truths. An
  entry is written for the robots from them.
- **Never names the robots, an issue number, a PR or a file** in the
  entry.
- Never edits the main checkouts (`~/pluggybot`, `~/rooftop-media-2026`):
  Ben runs tests and missions there. Work in a worktree
  (`~/pluggybot-worktrees/release-<date>`), created with `git -C`, one at a
  time.

## 1. Is there anything to release?

```sh
git -C ~/pluggybot fetch -q origin && git -C ~/rooftop-media-2026 fetch -q origin
git -C ~/pluggybot rev-list --count origin/main..origin/staging
git -C ~/rooftop-media-2026 rev-list --count origin/main..origin/staging
```

Both 0: nothing to release; say so and stop. Either repo may be 0 alone: then
it gets no PR, and the notes of the other say so. Check for a release PR
already open (`gh api 'repos/benholland1024/<repo>/pulls?state=open&base=main'`):
if one is, update its body (`gh api -X PATCH .../pulls/N -F body=@file`)
rather than opening another. Use `gh api` throughout: `gh pr edit` and
`gh issue view` fail on these repos.

## 2. Gather what is on staging

For each repo, every PR merged into staging since `main`, and any commit
that came without one:

```sh
git -C <repo> log --merges --first-parent origin/main..origin/staging --format='%h %s'
git -C <repo> log --no-merges --first-parent origin/main..origin/staging --format='%h %s'
gh api repos/benholland1024/<repo>/pulls/<N> --jq '.title, .body'
```

Read each PR's body and the issue it closes. In pluggybot read the
Observatory periods the release opens (`git -C ~/pluggybot diff
origin/main origin/staging -- docs/Observatory.md`): each period's "What
changed in the world" is the best source for the robots' entry, and its
"What the period is for" is exactly what the entry must NOT carry.

## 3. The release notes, for people

One body per PR, in this shape:

- **What is in it**, grouped (the world and the body; the mind and what
  the robots are told; the site; measurement and instruments; fixes;
  internal), a line per PR: its number, its title, what it changed in a
  sentence.
- **What the robots are told** (pluggybot's only): the changelog entry as
  they will read it, quoted whole, and that Ben approves it here.
- **Before merging**: anything that needs Ben (open calls, a box step such
  as `install.sh` or an `.env` change, a fixture re-vendored, a migration,
  a protocol change), and the order: the site's PR first, then the sim's.
- **After deploying**: what to read on the box (the sim's log, the header's
  commit, the period it opens in docs/Observatory.md).
- End with: 🤖 Generated with [Claude Code](https://claude.com/claude-code)

## 4. The changelog entry, for the robots

**Which entry.** Compare the file on `main` with the file on `staging`
(`git show origin/main:src/pluggybot/mind/changelog.md`). An entry staging
has and main does not is this release's (written by hand, or by a PR):
work into it and re-date it. Otherwise add a new `## YYYY-MM-DD` at the
top. The date is the release's day, UTC (`date -u +%F`). Dates are unique
and newest first, so a second release on one day adds to that day's entry.

**What goes in**: what changed that a robot could notice or use.
- A room, a building, a station, a job, a part of its body, a sense, a
  verb or a field, a rule of the world, something it is now shown.
- A fault it could have noticed, once fixed (a `say` that was always
  refused, a look that never came back).
- In the robots' terms: "you", "your", the world's own names (the house,
  the research center, the mall, the park, the storeroom's counter), and
  the keys, verbs and fields in backticks as the prompt names them.
- One fact an item, present tense, at most 500 characters
  (`changelog.MAX_ITEM_CHARS`); a few items an entry.

**What never goes in** (`tests/test_changelog.py` reads every item):
- What we measure or why, how anything is graded, what a period is for, an
  observatory, our hypotheses, what we hope or expect they will do.
- A hidden truth: the mouse's nature (the disclosure line is all that is
  ever said of it, and any entry about the mouse is Ben's call), a prop's
  hidden draws, the held-out tasks, why the study room is there.
- Advice: what to do about a change, a sentence that opens on a verb, a
  threshold, a tactic, an event-map row.
- A robot's name, an issue number, a PR, a commit, a file, a branch.
- The quadruped's or the rover's words (dock, rack, lying, draw, tools):
  `test_humanoid_words.py`'s list.

**What is left out**: anything a robot cannot notice or use: refactors,
tests, docs, instruments and spikes, measurement code, the site's admin
pages, a fixture re-vendored. The site's own changes go in only where a
robot meets them (a renderer for its look). If nothing on staging is one a
robot could notice, there is no entry: say so to Ben, and the release goes
without one.

**Check it**, in the worktree:

```sh
uv run pytest -q -n0 tests/test_changelog.py tests/test_humanoid_words.py
uv run python -c "from pluggybot.mind import changelog as c; print('\n'.join(c.current().lines()))"
```

The second prints the lines exactly as `recall` shows them; put them in
pluggybot's PR body under "What the robots are told".

## 5. Commit the entry onto staging

In a worktree off `origin/staging`, one commit, the file alone:

```sh
git -C ~/pluggybot worktree add ~/pluggybot-worktrees/release-<date> -b release-<date> origin/staging
# write the entry, run the check above
git -C ~/pluggybot-worktrees/release-<date> add src/pluggybot/mind/changelog.md
git -C ~/pluggybot-worktrees/release-<date> commit -m "The changelog's entry for the release of <date> (#617)"
git -C ~/pluggybot-worktrees/release-<date> push origin HEAD:staging
```

A plain `git commit`: the repo's identity is already right; never `-c
user.email`. The push is a fast-forward or nothing: if staging moved,
rebase the one commit onto it and push again. If the push is refused,
open the commit as a PR into staging instead
(`gh api repos/benholland1024/pluggybot/pulls -f head=release-<date> -f
base=staging ...`) and tell Ben it merges before the release PR.

## 6. Open the PRs, the site's first

```sh
gh api repos/benholland1024/rooftop-media-2026/pulls -f title=Release -f head=staging -f base=main -F body=@site-notes.md
gh api repos/benholland1024/pluggybot/pulls -f title=Release -f head=staging -f base=main -F body=@sim-notes.md
```

Read each body back (`gh api .../pulls/N --jq .body`).

## 7. Stop for Ben

Report, and do nothing more:
- the two PR links, the site's first;
- the robots' entry as they will read it, and that it waits for his word;
- what needs him before merging, and the order: merge the site's, then the
  sim's;
- that nothing was merged or deployed.

If Ben asks for a change to the entry, commit it onto staging the same
way: the release PR follows staging. Remove the worktree once the release
has merged.
