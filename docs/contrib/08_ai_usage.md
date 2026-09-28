# Using AI tools

AI coding tools are welcome in `pySDC`, and its maintainers use them too: [`aidecl.yaml`](./../../aidecl.yaml)
declares which tools helped with the code and how, and the CI validates that declaration on every run.
A contribution made with AI help is reviewed like any other, and a few rules keep it that way.

## You are the author

- Understand every line you submit, and be able to explain it in the review. "The tool wrote it" is not an answer
  to a reviewer's question.
- Check numerical code against something you trust: a known convergence order, a reference solution, a result from
  a paper. Passing tests are not enough when the tool also wrote the tests.
- Never let a tool make a failing test pass by loosening a tolerance, skipping a case or changing the expected
  value, unless you can say why the old value was wrong.
- Keep pull requests focused. A generated refactoring across many files is hard to review, so open an issue first.

## Say what the tool did

- Keep the `Co-authored-by:` trailer that agents add to their commits, also when squashing. `aidecl.yaml` counts
  these commits.
- In the pull request's description, mention which parts were written with AI help, e.g. "tests generated,
  implementation by hand", so the reviewer knows where to look closer.
- [Git AI](https://github.com/git-ai-project/git-ai) records which lines an agent wrote, without changing your
  workflow. It is optional, but recommended if you contribute regularly.

## Issues, reviews and discussions

Check what a tool found before you post it. An issue or a review comment with an unverified claim costs the
maintainers the time to disprove it, so post only what you have reproduced, and say how.

## Licenses

`pySDC` is under the BSD 2-clause license. Do not submit code that a tool reproduced from a source with an
incompatible license, and do not paste code or data into a tool that you may not share.

:arrow_left: [Back to publishing a new release](./07_release_guide.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md)
