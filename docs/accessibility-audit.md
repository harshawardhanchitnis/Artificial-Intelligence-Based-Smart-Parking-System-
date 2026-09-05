# Frontend consistency and accessibility audit

The audit covers landing, dashboard, presentation, image analysis, advanced correction, video,
parking lots, history/detail, analytics, diagnostics, reports, and system routes.

Resolved systemic issues include CSS cascade order, unreadable dark actions in History,
Diagnostics, and Reports, low-contrast disabled actions, missing focus-visible indicators,
undersized controls, absent skip links, mobile navigation, overflow-safe tables, upload controls,
loading/error states, pressed/tab semantics, and readable badges. Shared Button, LinkButton, and
StatusBadge primitives define consistent primary, secondary, outline, destructive, disabled,
hover, active, and keyboard-focus states.

The user-facing title is one coherent bold product name. The landing route is separate from the
operational dashboard. User-facing offline-mode slogans and development/release labels are removed;
local processing is described only where it explains privacy or system behavior. Internal package,
API, release-verification, and VERSION metadata remain intact.
