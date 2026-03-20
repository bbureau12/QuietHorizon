# ADR-009: Static Site Deployment via GitHub Pages

**Status:** Accepted  
**Date:** 2026-03-18  
**Deciders:** QuietHorizon Maintainers  
**Tags:** deployment, frontend, docs, hosting

## Context

QuietHorizon now includes a static browser-based interface in `docs/` that allows users to upload audio and submit it to a remote inference API. This interface is distinct from the Streamlit frontend in `frontend/`, which requires a Python runtime and is better suited to local or server-hosted application deployment.

We need a simple, low-maintenance way to publish the static interface so it is publicly accessible from the repository without introducing another hosting platform or a custom build pipeline.

Any chosen solution must support:

- Hosting plain HTML, CSS, and JavaScript from the repository
- Automatic deployment on changes to the static site
- A clean separation between static frontend hosting and remote model inference
- Low operational overhead for maintainers

## Decision

1. Host the static frontend in `docs/` using GitHub Pages.

2. Deploy the site with a GitHub Actions workflow:
- `.github/workflows/pages.yml`

3. Treat the `docs/` site as a static client-only frontend:
- No server-side Python executes on GitHub Pages
- Audio analysis requests are sent from the browser to an external inference API

4. Keep the Streamlit app as a separate deployment path for richer interactive workflows:
- `frontend/` remains the primary Python-based web application
- `docs/` is the lightweight public-facing static interface

5. Document GitHub Pages as a supported publishing path in the main project README.

## Consequences

### Positive Consequences

- Public deployment becomes inexpensive and easy to maintain.
- The static site can be published directly from the repository with no separate hosting provider.
- Deployments happen automatically when `docs/` changes on `main`.
- The hosting model is simple and transparent for contributors.
- The separation between frontend delivery and inference execution is explicit.

### Negative Consequences

- GitHub Pages cannot run the Python inference stack, so the site depends on a separate live API.
- The browser client cannot safely hold secrets; the inference endpoint must be designed for public access.
- Cross-origin concerns such as CORS must be handled by the remote API.
- The project now has two frontend surfaces to maintain: Streamlit and the static site.
- Runtime configuration is limited compared with platforms that support server-side logic or environment injection.

## Alternatives Considered

### Streamlit-Only Deployment

Rejected because GitHub Pages cannot host Python applications, and the existing Streamlit app has a heavier runtime and deployment footprint than needed for a lightweight public landing page.

### Another Static Hosting Provider

Options such as Netlify or Vercel would also work for a static site, but they introduce another platform, extra configuration, and a split deployment surface outside GitHub. GitHub Pages is sufficient for the current needs and keeps the deployment path close to the repository.

### No Public Deployment Automation

Rejected because manual publishing would be easy to forget, harder to reproduce, and inconsistent with the repository's existing emphasis on documented, repeatable workflows.

## References

- `.github/workflows/pages.yml`
- `docs/index.html`
- `docs/app.js`
- `docs/style.css`
- `README.md`
- `docs/adr/002-streamlit-frontend-framework.md`
