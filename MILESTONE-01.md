# Milestone 01 — Foundation

## Apply this deliverable

Extract the ZIP contents directly into:

`D:\Projects\AI Based Smart Parking System`

The archive does not contain or replace `Technical Documents/`.

Then run:

```powershell
Set-Location -LiteralPath 'D:\Projects\AI Based Smart Parking System'
Copy-Item -LiteralPath '.env.example' -Destination '.env'
powershell -ExecutionPolicy Bypass -File '.\scripts\setup.ps1'
```

## Acceptance criteria

- Git remains on `milestone/01-foundation`.
- All seven dataset archives validate.
- Four backend tests pass.
- Frontend lint and type checking pass.
- Backend health returns `healthy`.
- Dashboard, Analyse, and System pages render.
- No dataset is extracted or committed.

After verification, create the first commit:

```powershell
git add --all -- . ':(exclude)Technical Documents/**'
git commit -m "feat: establish smart parking application foundation"
git push -u origin milestone/01-foundation
```
