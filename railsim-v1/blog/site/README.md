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

## See it, then publish

```bash
cd railsim-v1
envs/rail_ops/.venv/bin/python blog/site/build.py --serve                 # preview on http://127.0.0.1:4321
envs/rail_ops/.venv/bin/python blog/site/build.py --push bpHigh/RailSimEnv-Inkling -m "Edit the intro"   # private
envs/rail_ops/.venv/bin/python blog/site/build.py --push bpHigh/RailSimEnv-Inkling --public -m "Publish"
```

The private Space is visible to you when you are logged in to Hugging Face. Making it public is the `--public` flag
(or Settings → Change visibility on the Space). The build needs node and npm and a Hugging Face token that can write.

## Rebuild the figures

- **Charts** read `content/assets/data/*`, written from the committed results by `make_data.py` (the banner also needs
  a running RailSimEnv server, see its docstring). The charts themselves are `content/embeds/*.html` (D3).
- **Train graphs** (`content/assets/image/graph-*.webp`) are the environment's `/viewer/?figure=1` page.
- **3D videos** (`public/media/*.mp4`, `.webp` posters) are frames of the environment's `/twin/` replay with the
  follow and cab cameras, captured with a headless browser and joined with ffmpeg.
