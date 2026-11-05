# Security policy

av-perception-lab is a research prototype. It has no releases: only the `main` branch is
maintained, and fixes go there.

## Reporting a vulnerability

Please do not open a public issue with exploit details.

1. Use GitHub's private vulnerability reporting: on the repository page, open **Security**,
   then **Report a vulnerability**.
2. If that button is not there, open a public issue that says only that you have a security
   report and asks for a private channel. Leave the details out.

Include the version (commit) you tested, what you did, what you expected and what happened.
You can expect a first answer within about two weeks; this is a one-person project.

## What is in scope

- The web interface (`av-perception-lab-web`): path handling for uploads, imports and run
  folders, the checks on `source.video` and `source.mcap`, and the host-header check.
- The file readers that take untrusted input: the MCAP reader, the QuickTime metadata
  parser, the ffprobe and ffmpeg calls on uploaded videos.
- The model download code: it must only fetch from addresses declared in the code.

## What is not

- The web interface has no authentication by design. It listens on `127.0.0.1` by default;
  exposing it with `--host 0.0.0.0` is your decision and your risk (see the security notes
  in `docs/reference.md`).
- Vulnerabilities in the model weights, or in third-party packages, belong to those projects.
  Tell us if our pinned version needs updating.
