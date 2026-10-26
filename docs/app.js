// Second Look front end. Every number shown comes from data/secondlook.json (built by
// scripts/build_app_data.py) or, in local mode, from the local server. This file
// only formats and adds; it never estimates.
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const state = { data: null, menu: null, path: "q", answers: {}, person: null, local: false, current: "" };
  const HEALTH = { 1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor" };

  const money = (v) => {
    const r = Math.round(v / 10) * 10;
    return (r < 0 ? "−$" : "$") + Math.abs(r).toLocaleString("en-US");
  };
  const exact = (v) => "$" + Math.round(v).toLocaleString("en-US");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  async function init() {
    state.data = await (await fetch("data/secondlook.json")).json();
    // Local mode is the make app server on port 8765; anywhere else (GitHub Pages) skip the probe.
    if (location.port === "8765") {
      try {
        const r = await fetch("api/health", { signal: AbortSignal.timeout(800) });
        state.local = r.ok;
      } catch { state.local = false; }
    }
    if (state.local) {
      $("#mode").textContent = "Local mode: worked out on this computer";
      $("#mode").classList.add("local");
      $("#local").classList.remove("hidden");
    }
    renderMenus();
    renderQuestions();
    renderPeople();
    renderSure();
    bindTabs();
    bindDialog();
    bindLocal();
  }

  function renderMenus() {
    const d = state.data;
    $("#menus").innerHTML = d.order.map((id) => {
      const m = d.menus[id];
      const prem = m.plans.map((p) => p.premium_monthly);
      return `<label class="row-choice"><input type="radio" name="menu" value="${id}">
        <div class="t">${esc(m.name)}</div>
        <div class="d">${m.plans.length} plans · you pay <span class="num">${exact(Math.min(...prem))}–${exact(Math.max(...prem))}</span> a month</div></label>`;
    }).join("");
    $("#menus").addEventListener("change", (e) => { state.menu = e.target.value; maybeRun(); });
  }

  function renderQuestions() {
    const qs = state.data.questions;
    $("#questions").innerHTML = qs.map((q) => `<fieldset class="q"><legend>${esc(q.text)}</legend><div class="opts">${
      q.options.map((o, i) => `<label class="opt"><input type="radio" name="${q.key}" value="${i}"><span>${esc(o)}</span></label>`).join("")
    }</div></fieldset>`).join("");
    const sync = () => {
      qs.forEach((q) => { const c = $(`input[name="${q.key}"]:checked`); state.answers[q.key] = c ? Number(c.value) : null; });
      $("#go").disabled = !state.menu || qs.some((q) => state.answers[q.key] == null);
    };
    $("#questions").addEventListener("change", sync);
    $("#menus").addEventListener("change", sync);
    sync();
    $("#go").addEventListener("click", () => { state.path = "q"; run(); });
  }

  function renderPeople() {
    $("#people").innerHTML = state.data.people.map((p) => `<label class="row-choice"><input type="radio" name="person" value="${p.id}">
      <div class="t">${esc(p.label)}</div>
      <div class="d">${p.age}, ${p.sex}, ${HEALTH[p.health]} health, ${p.conditions === 0 ? "no ongoing conditions" : p.conditions + " ongoing conditions"},
      ${p.rx_fills_2023} prescription fills and <span class="num">${exact(p.spend_2023)}</span> of care in 2023 · MEPS person ${p.id}</div></label>`).join("");
    $("#people").addEventListener("change", (e) => { state.person = e.target.value; state.path = "p"; maybeRun(); });
  }

  function bindTabs() {
    const tabs = [$("#tab-q"), $("#tab-p")];
    const select = (t) => {
      tabs.forEach((x) => { const on = x === t; x.setAttribute("aria-selected", on); x.tabIndex = on ? 0 : -1;
        $("#" + x.getAttribute("aria-controls")).classList.toggle("hidden", !on); });
      state.path = t.id === "tab-q" ? "q" : "p";
    };
    tabs.forEach((t, i) => {
      t.addEventListener("click", () => select(t));
      t.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = tabs[(i + 1) % 2]; select(n); n.focus(); }
      });
    });
    select(tabs[0]);
  }

  function maybeRun() { if (state.menu && ((state.path === "p" && state.person) || state.path === "q" && !$("#go").disabled && $("#result-card").dataset.shown)) run(); }

  async function run() {
    const d = state.data, menu = d.menus[state.menu];
    let res, who = null;
    if (state.path === "p") {
      who = d.people.find((p) => p.id === state.person);
      res = who.menus[state.menu];
    } else if (state.local) {
      const r = await fetch("api/answers", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ menu: state.menu, answers: d.questions.map((q) => state.answers[q.key]) }) });
      res = await r.json();
    } else {
      res = d.grid[state.menu][d.questions.map((q) => state.answers[q.key]).join("")];
    }
    renderResult(menu, res, who);
  }

  function renderResult(menu, res, who) {
    const order = menu.plans.map((p, i) => i).sort((a, b) => menu.plans[a].premium_monthly - menu.plans[b].premium_monthly);
    const lazyIdx = menu.plans.findIndex((p) => p.id === menu.lowest_premium);
    const lazy = menu.plans[lazyIdx];
    const max = Math.max(...res.bad, ...res.typical, 1);
    const verdict = res.flag
      ? `<p class="verdict closer">Look closer this year: your costs are hard to predict.</p>
         <p class="lede">Here is a typical year and a bad year (1 in 10) for every plan. People like you vary a lot, so Second Look does not name a winner.</p>`
      : `<p class="verdict lazy">You can be lazy: <em>${esc(lazy.name)}</em> (the lowest premium) is right for people like you.</p>
         <p class="lede">In a typical year it costs you about <span class="money">${money(res.typical[lazyIdx])}</span>; in a bad year (1 in 10), about <span class="money">${money(res.bad[lazyIdx])}</span>.</p>`;
    const rows = order.map((i) => {
      const p = menu.plans[i];
      const facts = p.facts.map((f, j) => `<dt>${esc(f.label)}</dt><dd><button type="button" class="quote-num" data-plan="${i}" data-fact="${j}"
          aria-label="${esc(f.label)} ${esc(f.value)}: show the source line">${esc(f.value)}</button></dd>`).join("");
      return `<div class="plan">
        <div class="plan-head"><span class="plan-name">${esc(p.name)}${i === lazyIdx ? '<span class="tag">lowest premium</span>' : ""}</span>
          <span class="plan-sub"><span class="num">${exact(p.premium_monthly * 1)}</span>/mo · <span class="num">${exact(p.deductible)}</span> deductible</span></div>
        <div class="bars">
          <div class="barline"><div class="track"><div class="fill typ" style="width:${Math.max(0, res.typical[i]) / max * 100}%"></div></div>
            <span class="v">${money(res.typical[i])}</span></div>
          <div class="barline"><div class="track"><div class="fill bad" style="width:${Math.max(0, res.bad[i]) / max * 100}%"></div></div>
            <span class="v bad">${money(res.bad[i])}</span></div>
        </div>
        <details class="facts"><summary>Plan numbers and where they come from</summary><dl class="fact-list">${facts}</dl></details>
      </div>`;
    }).join("");
    const current = res.flag ? "" : `<div class="note"><label for="current">Your plan this year (optional)</label><br>
      <select id="current"><option value="">Choose your current plan</option>${menu.plans.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("")}</select>
      <p id="delta" aria-live="polite"></p></div>`;
    const actual = who ? actualTable(menu, who.menus[state.menu], order) : "";
    $("#result-card").innerHTML = `<div class="fade">${verdict}
      <div class="legend"><span><span class="sw typ"></span>Typical year</span><span><span class="sw bad"></span>Bad year (1 in 10)</span>
        <span>12 months of premiums plus your share of care, less any employer HSA deposit</span></div>
      ${rows}${current}${actual}
      <p class="mute" style="font-size:0.85rem;margin-top:14px">How these are worked out: a typical year is what each plan would have cost the five survey participants whose actual 2024 spending was nearest to ${who ? "this person’s" : "your"} predicted median${who || res.typical_spend ? ` (<span class="num">${exact(who ? who.predicted_typical_spend : res.typical_spend)}</span> of care)` : ""}; a bad year uses the predicted 90th percentile${who || res.bad_spend ? ` (<span class="num">${exact(who ? who.predicted_bad_spend : res.bad_spend)}</span>)` : ""}. Tap any plan number to see the line it came from.</p></div>`;
    $("#result-card").dataset.shown = "1";
    $("#result-card").querySelectorAll("button.quote-num").forEach((b) => b.addEventListener("click", () => openQuote(menu, b)));
    const sel = $("#current");
    if (sel) sel.addEventListener("change", () => {
      if (sel.value === "") { $("#delta").textContent = ""; return; }
      const dlt = res.typical[Number(sel.value)] - res.typical[lazyIdx];
      $("#delta").innerHTML = Number(sel.value) === lazyIdx ? "Your current plan is the lowest-premium plan. Same as last year is fine."
        : Math.abs(dlt) < 10 ? "Same as last year would cost about the same in a typical year."
        : dlt > 0 ? `Same as last year would cost about <span class="money">${money(dlt)}</span> more in a typical year.`
        : `Same as last year would cost about <span class="money">${money(-dlt)}</span> less in a typical year, so keeping it is fine.`;
    });
    $("#result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }

  function actualTable(menu, r, order) {
    const best = Math.min(...r.actual_2024);
    const pick = menu.plans.find((p) => p.id === r.last_year_pick);
    return `<div class="note"><h3>What actually happened in 2024</h3>
      <p class="mute" style="margin-top:4px">This person’s real 2024 care, priced under each 2027 plan. Their 2023 care pointed to ${esc(pick.name)}.</p>
      <table class="actual"><thead><tr><th scope="col">Plan</th><th scope="col" style="text-align:right">2024 care, priced</th></tr></thead><tbody>${
        order.map((i) => `<tr><td>${esc(menu.plans[i].name)}${r.actual_2024[i] === best ? ' <span class="tag">cheapest</span>' : ""}</td>
          <td class="money">${money(r.actual_2024[i])}</td></tr>`).join("")}</tbody></table></div>`;
  }

  let opener = null;
  function bindDialog() {
    const dlg = $("#quote");
    $("#q-close").addEventListener("click", () => dlg.close());
    dlg.addEventListener("close", () => opener && opener.focus());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  }

  function sourceBlock(s) {
    return `<blockquote class="src">…${esc(s.before)} <mark>${esc(s.quote)}</mark> ${esc(s.after)}…</blockquote>`;
  }

  function openQuote(menu, btn) {
    opener = btn;
    const plan = menu.plans[Number(btn.dataset.plan)], f = plan.facts[Number(btn.dataset.fact)], s = f.source;
    $("#q-title").textContent = `${f.label}: ${f.value}`;
    $("#q-meta").textContent = `${s.doc === plan.id ? s.title : plan.name + " · " + s.title} · page ${s.page}`;
    $("#q-text").outerHTML = sourceBlock(s).replace('class="src"', 'class="src" id="q-text"');
    const extra = [];
    if (s.second) extra.push(`<p class="src-meta">Your share comes from ${esc(s.second.title)}, page ${s.second.page}:</p>${sourceBlock(s.second)}`);
    if (s.note) extra.push(`<p>${esc(s.note)}</p>`);
    extra.push(s.provenance.startsWith("transcribed")
      ? `<p class="check"><strong>Check this.</strong> Second Look’s automatic reading got this value wrong, so this line was found by hand and checked against the PDF.</p>`
      : `<p class="src-meta">${s.provenance.startsWith("read") ? "Read from the PDF and checked against its text." : "From the " + esc(s.provenance) + ", checked against its text."}</p>`);
    extra.push(`<p><a href="${esc(s.url)}" target="_blank" rel="noopener">Open the original document</a></p>`);
    $("#q-extra").innerHTML = extra.join("");
    $("#quote").showModal();
    $("#q-close").focus();
  }

  function renderSure() {
    const t = state.data.trust, pc = (x) => Math.round(x * 100) + "%";
    $("#sure-body").innerHTML = `<ul>
      <li>We tested the lazy rule on <span class="big">${t.people}</span> real adults from the US Medical Expenditure Panel Survey, against these <span class="big">${t.menus}</span> menus, using their actual 2024 care. Taking the lowest premium gave the cheapest plan to <span class="big">${pc(t.lowest_premium_hit)}</span> of them; picking what was best last year did so for <span class="big">${pc(t.same_as_last_year_hit)}</span>. Average overpayment: <span class="big">${exact(t.lowest_premium_regret)}</span> against <span class="big">${exact(t.same_as_last_year_regret)}</span>.</li>
      <li>The lowest premium cost more than $250 extra in about <span class="big">${pc(t.lazy_costly_share)}</span> of cases. Second Look’s check ranks those cases first with an AUC of <span class="big">${t.flagger_auc.toFixed(2)}</span> (95% interval ${t.flagger_auc_ci[0].toFixed(2)} to ${t.flagger_auc_ci[1].toFixed(2)}) and catches about <span class="big">${pc(t.flagger_recall)}</span> of them. The five-question version knows less about you: AUC <span class="big">${t.five_answer_flagger_auc.toFixed(2)}</span>. When it says “look closer”, it does not name a plan, because choosing for flagged people did not beat the lowest premium in testing.</li>
      <li>Plan numbers are read from the plan PDFs. On 20 held-out 2027 plan documents, <span class="big">${pc(t.reader_accuracy)}</span> of ${t.reader_values} key values were right, <span class="big">${(t.reader_wrong * 100).toFixed(1)}%</span> were wrong and <span class="big">${(t.reader_needs_check * 100).toFixed(1)}%</span> were flagged for checking. Values read wrong are shown here with the correct line and a “check this” note.</li>
    </ul>
    <h3>What these estimates leave out or simplify</h3><ul>${t.approximations.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>
    <p class="mute" style="font-size:0.85rem">Sources: ${t.sources.map(esc).join(", ")} in the project repository.</p>`;
  }

  function bindLocal() {
    const input = $("#pdf"), btn = $("#read");
    input.addEventListener("change", () => { btn.disabled = !input.files.length; });
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      $("#read-out").innerHTML = `<p class="mute">Reading on this computer. This takes about two minutes…</p>`;
      const fd = new FormData(); fd.append("pdf", input.files[0]);
      const r = await (await fetch("api/read", { method: "POST", body: fd })).json();
      $("#read-out").innerHTML = `<dl class="fact-list" style="margin-top:14px">${r.fields.map((f) => `<dt>${esc(f.label)}</dt><dd>${
        f.status === "verified" ? `<span class="num">${esc(f.display)}</span><br><span class="mute" style="font-size:0.8rem">page ${f.page}: <mark>${esc(f.quote)}</mark></span>`
        : `<span class="check"><strong>Check this.</strong> ${esc(f.reason)}</span>`}</dd>`).join("")}</dl>
        <p class="mute" style="font-size:0.85rem">${r.seconds} s on this computer. Nothing was sent anywhere.</p>`;
      btn.disabled = false;
    });
  }

  init();
})();
