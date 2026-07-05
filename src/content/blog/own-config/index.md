---
title: "My Work Setup & Config"
description: "A rundown of the tools, themes, and workflows I use day-to-day — maybe useful if you're curious about what I use."
date: 2026-06-15
authors:
  - maokaihe
image: ./assets/banner.png
tags:
  - config
---

## [VS Code](https://code.visualstudio.com/)

I like this very much, because it has a ton of plugins for coding and customization.

I prefer the [Everforest Dark](https://github.com/sainnhe/everforest) theme with [Menlo](https://en.wikipedia.org/wiki/Menlo_(typeface)) ,[Iosevka](https://typeof.net/Iosevka/) and [Jetbrains Mono](https://www.jetbrains.com/lp/mono/)fonts.

I like to move the left sidebar to the top-left (it's on the left by default).

Some other themes I've used and can recommend:
- [Catppuccin](https://catppuccin.com/)
- [Gruvbox](https://github.com/morhetz/gruvbox)
- [Tokyonight](https://github.com/enkia/tokyo-night-vscode-theme)

### Rust

Just install [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer), and optionally add [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) and [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb) for a better experience. That's pretty much all you need.

### Vue

The [Vue - Official](https://marketplace.visualstudio.com/items?itemName=Vue.volar) plugin plus [Biome](https://biomejs.dev/) for linting. These days I write frontend code primarily with AI agents — I have to say, AI is really good at building UI.

As for package managers, I like to use [pnpm](https://pnpm.io/) and [Bun](https://bun.sh/). That said, Bun was recently rewritten in Rust, so I feel like it might not be fully stable yet.

### Python

I prefer to use [Ruff](https://docs.astral.sh/ruff/) and [Ty](https://docs.astral.sh/ty/) for writing Python — both are built by [Astral](https://astral.sh/) in Rust.

Ruff is way faster than other tools, I'm sure you'll like it.

## [Zed](https://zed.dev/)

I only used Zed for a short time because it's not quite as smooth as VS Code for me. I've been following it since very early on, and even though it's reached 1.0 now, it still has some usability and ergonomics issues. But I hope it keeps getting better over the next year.

## Input Method

I use [Xiaohe Shuangpin](https://en.wikipedia.org/wiki/Shuangpin) (a double-pinyin input scheme) for typing Chinese, which I started learning in 2024. Honestly it's not hard to pick up — you just need a day to memorize the mapping table, and after about two weeks of practice you'll be pretty fluent. The reason I switched to double-pinyin is that I used to mistype a lot with regular full pinyin.

## Linux

Even though I use a Mac now, before I got one I spent about half a year using Linux as my daily driver. I came across [Omarchy](https://omarchy.org/) at the time and gave it a try — it was honestly great. It's based on [Arch Linux](https://archlinux.org/) and uses [Hyprland](https://hyprland.org/) as its window manager, which looks really nice.

Once I switched to Arch I realized installing software is actually super easy thanks to the massive and comprehensive [AUR](https://aur.archlinux.org/) packages. I used it for half a year and never had a single breakage — feels pretty stable to me.

That said, if you want rock-solid stability, you're better off with [Linux Mint](https://linuxmint.com/), [Ubuntu](https://ubuntu.com/), or [Fedora](https://fedoraproject.org/). I'd personally recommend Fedora and Mint — I'm just not a big fan of Ubuntu.

## CLI Tools

Since I'm really into Rust these days, a lot of the CLI tools I use are written in Rust, though some Go and C++ ones are great too.

- [eza](https://github.com/eza-community/eza) — modern replacement for `ls`
- [rg](https://github.com/BurntSushi/ripgrep) — fast grep alternative
- [bat](https://github.com/sharkdp/bat) — cat clone with syntax highlighting
- [fd](https://github.com/sharkdp/fd) — simple, fast find alternative
- [starship](https://starship.rs/) — cross-shell prompt
- [btop](https://github.com/aristocratos/btop) — resource monitor
- [zoxide](https://github.com/ajeetdsouza/zoxide) — smarter cd command
- [fzf](https://github.com/junegunn/fzf) — fuzzy finder
- [broot](https://github.com/Canop/broot) — interactive tree explorer
- [yazi](https://yazi-rs.github.io/) — terminal file manager
- [lazygit](https://github.com/jesseduffield/lazygit) — terminal UI for git

For Mac:
- [mole](https://github.com/tw93/mole) — Clean, uninstall, analyze, optimize, and monitor your Mac 

For Node version management:
- [fnm](https://github.com/Schniz/fnm) — written in Rust, easy to use

As for terminal emulators, I've tried [Alacritty](https://alacritty.org/), [Kitty](https://sw.kovidgoyal.net/kitty/), and [Ghostty](https://ghostty.org/). They're all pretty similar for my use case, but I lean towards Kitty a bit more.

---

Update:

Now I find `Qoder`, which is currently good for agent coding. I feel it is close to `Cursor`. I use `Qoder` for two weeks and use its cheap model `Qwen3.7-Max`, even though not good as gpt5.5 and claude, it can finish some medium work with much less cost.

