/* Fenn Dashboard — client-side interactivity */

(function () {
  "use strict";

  // ── Log viewer filters ─────────────────────────────────────────────────── //

  const levelState = { info: true, warning: true, exception: true };
  const kindState  = { system: true, user: true, print: true };

  function applyFilters() {
    const query = (document.getElementById("log-search")?.value || "").toLowerCase().trim();

    document.querySelectorAll(".log-entry").forEach((row) => {
      const level = row.dataset.level;
      const kind  = row.dataset.kind;
      const msg   = row.querySelector(".log-msg")?.textContent || "";

      const levelOk = levelState[level] !== false;
      const kindOk  = kindState[kind]   !== false;
      const searchOk = !query || msg.toLowerCase().includes(query);

      if (levelOk && kindOk && searchOk) {
        row.classList.remove("hidden", "search-hidden");
      } else {
        row.classList.add("hidden");
      }
    });

    // Highlight search matches — DOM-only to avoid XSS via innerHTML
    if (query) {
      document.querySelectorAll(".log-entry:not(.hidden) .log-msg").forEach((el) => {
        const text = el.textContent;
        while (el.firstChild) el.removeChild(el.firstChild);
        const regex = new RegExp(escapeRegex(query), "gi");
        let last = 0, match;
        while ((match = regex.exec(text)) !== null) {
          if (match.index > last) el.appendChild(document.createTextNode(text.slice(last, match.index)));
          const span = document.createElement("span");
          span.className = "highlight";
          span.textContent = match[0];
          el.appendChild(span);
          last = match.index + match[0].length;
        }
        if (last < text.length) el.appendChild(document.createTextNode(text.slice(last)));
      });
    } else {
      document.querySelectorAll(".log-msg .highlight").forEach((span) => {
        span.replaceWith(document.createTextNode(span.textContent));
      });
    }

    updateCount();
  }

  function updateCount() {
    const total   = document.querySelectorAll(".log-entry").length;
    const visible = document.querySelectorAll(".log-entry:not(.hidden)").length;
    const el = document.getElementById("log-visible-count");
    if (el) el.textContent = visible === total ? `${total} entries` : `${visible} / ${total} entries`;
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // Level filter buttons
  document.querySelectorAll("[data-filter-level]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const lvl = btn.dataset.filterLevel;
      if (lvl === "all") {
        Object.keys(levelState).forEach((k) => (levelState[k] = true));
        document.querySelectorAll("[data-filter-level]").forEach((b) => {
          b.classList.remove("active-info", "active-warning", "active-exception");
        });
        btn.classList.add("active-all");
      } else {
        levelState[lvl] = !levelState[lvl];
        document.querySelector("[data-filter-level='all']")?.classList.remove("active-all");
        btn.classList.toggle(`active-${lvl}`, levelState[lvl]);
      }
      applyFilters();
    });
  });

  // Kind filter buttons
  document.querySelectorAll("[data-filter-kind]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const k = btn.dataset.filterKind;
      if (k === "all") {
        Object.keys(kindState).forEach((key) => (kindState[key] = true));
        document.querySelectorAll("[data-filter-kind]").forEach((b) => {
          b.classList.remove("active-system", "active-user", "active-print");
        });
        btn.classList.add("active-all");
      } else {
        kindState[k] = !kindState[k];
        document.querySelector("[data-filter-kind='all']")?.classList.remove("active-all");
        btn.classList.toggle(`active-${k}`, kindState[k]);
      }
      applyFilters();
    });
  });

  // Search input
  const logSearch = document.getElementById("log-search");
  if (logSearch) {
    logSearch.addEventListener("input", applyFilters);
    logSearch.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { logSearch.value = ""; applyFilters(); }
    });
  }

  // ── Global session/project search (index page) ─────────────────────────── //

  const globalSearch = document.getElementById("global-search");
  if (globalSearch) {
    globalSearch.addEventListener("input", () => {
      const q = globalSearch.value.toLowerCase().trim();
      document.querySelectorAll("[data-searchable]").forEach((el) => {
        const text = el.dataset.searchable.toLowerCase();
        el.style.display = !q || text.includes(q) ? "" : "none";
      });
    });
  }

  // ── Session search (index page) ────────────────────────────────────────── //

  const sessionSearch = document.getElementById("session-search");
  if (sessionSearch) {
     sessionSearch.addEventListener("input", () => {
    const q = sessionSearch.value.toLowerCase().trim();
    const rows = document.querySelectorAll("tr[data-session]");

    rows.forEach((row) => {
      const cells = row.querySelectorAll("td[data-searchable]");
      let match = false;

      cells.forEach((cell) => {
        const text = cell.getAttribute("data-searchable").toLowerCase();
        if (text.includes(q)) {
          match = true;
        }
      });

      row.style.display = match ? "" : "none";
    });
  });
  }

  // ── Logout confirmation dialog ─────────────────────────────────────────── //

  const logoutForm    = document.getElementById("logout-form");
  const logoutDialog  = document.getElementById("logout-dialog");
  const logoutCancel  = document.getElementById("logout-cancel");
  const logoutConfirm = document.getElementById("logout-confirm");

  if (logoutForm && logoutDialog) {
    const fallbackPrompt =
      "Disconnect? You'll need to paste your dashboard token again next time. " +
      "If you just want to stop, close this tab instead.";

    logoutForm.addEventListener("submit", (e) => {
      // If the user already confirmed via the dialog, the form has been flagged
      // and we let the native submit proceed.
      if (logoutForm.dataset.confirmed === "true") return;
      e.preventDefault();
      if (typeof logoutDialog.showModal === "function") {
        logoutDialog.showModal();
      } else if (window.confirm(fallbackPrompt)) {
        logoutForm.dataset.confirmed = "true";
        logoutForm.submit();
      }
    });

    logoutCancel?.addEventListener("click", () => logoutDialog.close());

    logoutConfirm?.addEventListener("click", () => {
      logoutForm.dataset.confirmed = "true";
      logoutDialog.close();
      logoutForm.submit();
    });

    // Clicking the backdrop (outside the dialog content) also cancels.
    logoutDialog.addEventListener("click", (e) => {
      if (e.target === logoutDialog) logoutDialog.close();
    });
  }

  // ── Collapsible sections ───────────────────────────────────────────────── //

  document.querySelectorAll(".collapsible-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = document.getElementById(btn.dataset.target);
      if (!target) return;
      const collapsed = target.classList.toggle("collapsed");
      const arrow = btn.querySelector(".arrow");
      if (arrow) arrow.textContent = collapsed ? "▶" : "▼";
      btn.setAttribute("aria-expanded", String(!collapsed));
    });
  });

  // ── Auto-refresh for running sessions ──────────────────────────────────── //

  const sessionStatus = document.getElementById("session-status");
  if (sessionStatus && sessionStatus.dataset.status === "running") {
    const project   = sessionStatus.dataset.project;
    const sessionId = sessionStatus.dataset.session;

    function createEntryEl(entry) {
      const row = document.createElement("div");
      row.className = `log-entry level-${entry.level} kind-${entry.kind}`;
      row.dataset.level = entry.level;
      row.dataset.kind  = entry.kind;

      const ts = document.createElement("span");
      ts.className = "log-ts";
      ts.textContent = entry.ts;

      const kindWrap = document.createElement("span");
      kindWrap.className = "log-kind";
      const kindBadge = document.createElement("span");
      kindBadge.className = `badge badge-${entry.kind}`;
      kindBadge.style.fontSize = "10px";
      kindBadge.textContent = entry.kind;
      kindWrap.appendChild(kindBadge);

      const levelWrap = document.createElement("span");
      levelWrap.className = "log-level";
      const levelBadge = document.createElement("span");
      levelBadge.className = `badge badge-${entry.level}`;
      levelBadge.style.fontSize = "10px";
      levelBadge.textContent = entry.level;
      levelWrap.appendChild(levelBadge);

      const msg = document.createElement("span");
      msg.className = "log-msg";
      msg.textContent = entry.message;

      row.append(ts, kindWrap, levelWrap, msg);
      return row;
    }

    function refresh() {
      fetch(`/api/session/${encodeURIComponent(project)}/${encodeURIComponent(sessionId)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (!data) return;

          // Status changed (e.g. running → completed/crashed) — full reload
          // because the header badge and meta row need server-side re-render.
          if (data.status !== "running") {
            location.reload();
            return;
          }

          // New entries — append without disturbing filters or scroll position.
          const currentCount = document.querySelectorAll(".log-entry").length;
          if (data.entry_count !== currentCount && Array.isArray(data.entries)) {
            const container = document.querySelector(".log-entries");
            if (!container) return;
            // Remove the "no entries" empty-state placeholder if present.
            const empty = container.querySelector(".empty");
            if (empty) empty.remove();
            data.entries.slice(currentCount).forEach((entry) => {
              const row = createEntryEl(entry);
              // Respect current filter state so new rows appear/hide correctly.
              if (levelState[entry.level] === false || kindState[entry.kind] === false) {
                row.classList.add("hidden");
              }
              container.appendChild(row);
            });
            updateCount();
          }
        })
        .catch(() => {});
    }

    setInterval(refresh, 5000);
  }

  // ── Keyboard shortcuts ─────────────────────────────────────────────────── //

  document.addEventListener("keydown", (e) => {
    // Ctrl/Cmd + K → focus search
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      const s = document.getElementById("log-search") || document.getElementById("global-search");
      if (s) { s.focus(); s.select(); }
    }
    // Escape → blur search
    if (e.key === "Escape") {
      document.activeElement?.blur();
    }
  });
  // ── Sidebar hamburger toggle ───────────────────────────────────────────── //

  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebar = document.getElementById("sidebar");

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", () => {
      const isOpen = sidebar.classList.toggle("open");
      sidebarToggle.classList.toggle("is-open", isOpen);
    });

    // Close sidebar when clicking a nav item on mobile
    sidebar.querySelectorAll(".nav-item").forEach((item) => {
      item.addEventListener("click", () => {
        if (window.innerWidth <= 600) {
          sidebar.classList.remove("open");
          sidebarToggle.classList.remove("is-open");
        }
      });
    });

    // Close sidebar when clicking outside
    document.addEventListener("click", (e) => {
      if (window.innerWidth <= 600 &&
          sidebar.classList.contains("open") &&
          !sidebar.contains(e.target) &&
          !sidebarToggle.contains(e.target)) {
        sidebar.classList.remove("open");
        sidebarToggle.classList.remove("is-open");
      }
    });
  }

  // ── Init ───────────────────────────────────────────────────────────────── //

  updateCount();

})();
