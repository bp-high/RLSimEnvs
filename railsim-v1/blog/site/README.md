# The Inkling 2 Small article (a Hugging Face Space)

The article on reinforcement learning with Inkling 2 Small on RailSimEnv, built with Hugging Face's
[research article template](https://huggingface.co/spaces/tfrere/research-article-template) (CC BY 4.0) and published
as a static Space. It starts private; nothing is public until you say so.

## Edit the text

All the words are in `content/`, in Markdown (MDX: Markdown plus a few components):

| file | what is in it |
|---|---|
| `content/article.mdx` | title, subtitle, author, affiliation, date, the links under the title, licence line |
| `content/chapters/01-intro.mdx` | The job (with the hero video and the one-paragraph summary) |
| `content/chapters/02-environment.mdx` | The environment, the rules, the scoring, the results chart |
| `content/chapters/03-setup.mdx` | The model and the training loop |
| `content/chapters/04-round-64k.mdx` | Round 1 |
| `content/chapters/05-round-262k.mdx` | Round 2, the per-task chart, the learning curves |
| `content/chapters/06-on-the-line.mdx` | The three tasks, with train graphs and 3D videos |
| `content/chapters/07-lessons.mdx` | What we noticed |
| `content/chapters/08-what-it-took.mdx` | Tokens and time |
| `content/chapters/09-next.mdx` | What's next, links, credits |
| `content/bibliography.bib` | references, cited in the text as `[@key]` |

Things you can write:

- **Footnotes:** `a claim[^why]` and, anywhere in the same file, `[^why]: the footnote text.`
- **Citations:** `[@tinker2026]`; add new entries to `bibliography.bib`.
- **A margin note:** `<Sidenote>text</Sidenote>`; **a callout:** `<Note variant="info" title="...">text</Note>`.
- **An image:** put it in `content/assets/image/`, import it at the top of the chapter
  (`import pic from '../assets/image/pic.webp';`) and write `<Image src={pic} alt="..." caption="..." zoomable />`.
- **A video or GIF:** put it in `public/media/` and use the `<figure class="video-figure">` block from chapter 6.
- Headings `##` and `###` make the table of contents.

## Edit, review, publish

There are two private Spaces: a **draft** to review on any device, and the **real** one.
The draft Space can be deleted once a build is published: the next `--draft` creates it again (private).

| Space | address | who pushes to it |
|---|---|---|
| draft | [bpHigh/RailSimEnv-Inkling-draft](https://huggingface.co/spaces/bpHigh/RailSimEnv-Inkling-draft) | `--draft`, any time |
| real | [bpHigh/RailSimEnv-Inkling](https://huggingface.co/spaces/bpHigh/RailSimEnv-Inkling) | `--publish`, only a build already on the draft |

```bash
cd railsim-v1
envs/rail_ops/.venv/bin/python blog/site/build.py --dev                            # edit: live reload on :4321
envs/rail_ops/.venv/bin/python blog/site/build.py --diff                           # what changed since the real Space
envs/rail_ops/.venv/bin/python blog/site/build.py --draft -m "Edit the intro"      # push to the draft, review it there
envs/rail_ops/.venv/bin/python blog/site/build.py --publish -m "Edit the intro"    # same build to the real Space
envs/rail_ops/.venv/bin/python blog/site/build.py --publish --public -m "Publish"  # ... and make it public
```

1. **Edit** the files in `content/`. With `--dev` running, the page at http://127.0.0.1:4321 reloads on every save.
   From a Claude session, ask for the change in words ("shorten the intro", "add a footnote to round 2 saying ...");
   the session edits these files and the preview shows the result. You can also edit the `.mdx` files on GitHub on
   the working branch and ask the session to pull them.
2. **Review**: `--diff` lists the text changes, word by word, against the commit the real Space was built from.
   `--draft` pushes the build to the draft Space; open it while logged in to Hugging Face, on any device.
3. **Publish**: `--publish` rebuilds and pushes to the real Space, and refuses unless the draft Space has exactly the
   same build. Each push writes `build.json` (a fingerprint of everything that goes into the page, the git commit and
   whether there were uncommitted changes); the check compares fingerprints, so a change made after the draft was
   pushed has to go to the draft first. `--push REPO` pushes anywhere without the check.

`--serve` shows the exact build that would be pushed (scripts inlined) on :4321. The build needs node and npm and a
Hugging Face token that can write.

## Rebuild the figures

- **Charts** read `content/assets/data/*`, written from the committed results by `make_data.py` (the banner also needs
  a running RailSimEnv server, see its docstring). The charts themselves are `content/embeds/*.html` (D3).
- **Train graphs** (`content/assets/image/graph-*.webp`) are the environment's `/viewer/?figure=1` page.
- **3D videos** (`public/media/*.mp4`, `.webp` posters) are frames of the environment's `/twin/` replay with the
  follow and cab cameras, captured with a headless browser and joined with ffmpeg.
