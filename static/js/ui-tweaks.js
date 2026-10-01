/*
 * 界面增强层（主题 / 状态胶囊 / 表格可读性 / 问答开关）
 *
 * 与 index.html 解耦：只读取已有 DOM，不改动它的结构，
 * 所有函数都挂在 window.UITweaks 上供内联 onclick 调用。
 */
(function () {
    'use strict';

    const LS = {
        theme: 'fa-theme',
        llm: 'fa-use-llm',
        collapsed: 'fa-folded'
    };

    const $ = id => document.getElementById(id);
    const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

    // ================= 主题 =================
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const meta = document.querySelector('meta[name="theme-color"]');
        if (meta) meta.setAttribute('content', theme === 'dark' ? '#0d1117' : '#ffffff');
        retintCharts(theme === 'dark');
        const btn = $('theme-btn');
        if (btn) btn.textContent = theme === 'dark' ? '☀' : '◐';
        try { localStorage.setItem(LS.theme, theme); } catch (e) { /* 隐私模式忽略 */ }
    }

    function currentTheme() {
        const saved = safeGet(LS.theme);
        if (saved === 'dark' || saved === 'light') return saved;
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    /** ECharts 是 canvas 渲染，CSS 变量管不到它，切主题时要显式重涂文字与坐标轴。 */
    function retintCharts(dark) {
        if (!window.echarts) return;
        const axis = dark ? '#8d96a0' : '#5f6368';
        const line = dark ? '#262e39' : '#e0e0e0';
        $$('.chart-container').forEach(el => {
            const inst = echarts.getInstanceByDom(el);
            if (!inst) return;
            try {
                inst.setOption({
                    textStyle: { color: dark ? '#c9d1d9' : '#202124' },
                    legend: { textStyle: { color: axis } },
                    tooltip: { backgroundColor: dark ? '#1c232c' : '#fff', borderColor: line,
                               textStyle: { color: dark ? '#e6edf3' : '#202124' } },
                    xAxis: { axisLabel: { color: axis }, axisLine: { lineStyle: { color: line } },
                             splitLine: { lineStyle: { color: line } } },
                    yAxis: { axisLabel: { color: axis }, axisLine: { lineStyle: { color: line } },
                             splitLine: { lineStyle: { color: line } } }
                });
            } catch (e) { /* 某些图没有该轴配置，忽略 */ }
        });
    }

    function safeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
    function safeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { } }

    // ================= 顶栏状态胶囊 =================
    function ago(sec) {
        if (sec === null || sec === undefined) return '未加载';
        if (sec < 60) return Math.round(sec) + ' 秒前';
        if (sec < 3600) return Math.round(sec / 60) + ' 分钟前';
        if (sec < 86400) return (sec / 3600).toFixed(1) + ' 小时前';
        return (sec / 86400).toFixed(1) + ' 天前';
    }

    function setChip(id, text, cls, title) {
        const el = $(id);
        if (!el) return;
        el.className = 'chip ' + (cls || '');
        el.innerHTML = '<span class="dot"></span>' + text;
        if (title) el.title = title;
    }

    async function refreshStatus() {
        try {
            const resp = await fetch('/api/health');
            const h = await resp.json();
            setChip('chip-data', h.memory_cache_hit ? '数据已就绪' : '数据待加载',
                h.memory_cache_hit ? 'ok' : 'warn',
                '排名缓存新鲜度：' + ago(h.cache_age_seconds) + '；磁盘缓存存在：' + h.cache_exists);
            setChip('chip-breaker', h.breaker === 'open' ? '上游限流中' : '数据源正常',
                h.breaker === 'open' ? 'err' : 'ok',
                'akshare 接口熔断状态' + (h.open_breakers && h.open_breakers.length
                    ? '（开放：' + h.open_breakers.join(', ') + '）' : ''));
            setChip('chip-llm', h.llm_configured ? 'LLM：' + h.llm_model : '规则引擎模式',
                h.llm_configured ? 'ok' : '', h.llm_configured ? '已配置 LLM，可在问答面板关闭' : '未配置 API Key，全部走本地规则引擎');
            if (typeof syncLlmAvailability === 'function') syncLlmAvailability(h.llm_configured);
            return h;
        } catch (e) {
            setChip('chip-data', '后端不可达', 'err', String(e));
            return null;
        }
    }

    // ================= 表格可读性 =================
    const CODE_RE = /^\d{6}$/;
    const NUM_RE = /^[+-]?\d{1,3}(,\d{3})*(\.\d+)?%?$/;

    const TABLE_SELECTOR = 'table.data-table, table#funds-table, table#compare-table';

    function enhanceTables(root) {
        $$(TABLE_SELECTOR, root || document).forEach(table => {
            const heads = $$('th', table).map(th => (th.textContent || '').trim());
            $$('tbody tr:not([data-ench])', table).forEach(tr => {
                tr.setAttribute('data-ench', '1');
                $$('td', tr).forEach((td, i) => {
                    const head = heads[i] || '';
                    const txt = (td.textContent || '').trim();
                    if (!txt || td.querySelector('.spark')) return;
                    if (txt.length > 10 && !td.title) td.title = txt;   // 被省略号截断时可悬停看全文
                    if (CODE_RE.test(txt)) { td.classList.add('muted'); return; }   // 基金代码不着色
                    if (!NUM_RE.test(txt)) return;
                    td.classList.add('num');
                    const raw = txt.replace(/[,%]/g, '');
                    const v = parseFloat(raw);
                    if (isNaN(v) || v === 0) return;
                    if (/代码|名称|点数|次数|天数|日期/.test(head)) return;
                    const isPct = txt.includes('%');
                    if (isPct || /收益|回报|回撤|波动|超额|率/.test(head)) {
                        td.classList.add(v > 0 ? 'up' : 'down');
                        // 背景"量条"只做强弱提示，压到很淡，避免盖过数字本身
                        const w = Math.min(Math.abs(v), 60) / 60 * 26;
                        const color = v > 0 ? 'var(--color-up)' : 'var(--color-down)';
                        td.style.background = `linear-gradient(90deg, transparent ${100 - w}%, ` +
                              `color-mix(in srgb, ${color} 13%, transparent) ${100 - w}%)`;
                    }
                });
            });
        });
    }

    // ================= 问答：LLM 开关 =================
    let llmAvailable = false;

    function useLlm() {
        const box = $('llm-toggle');
        return box ? !!box.checked : false;
    }

    function setUseLlm(on, persist) {
        const box = $('llm-toggle');
        if (box) box.checked = !!on;
        if (persist !== false) safeSet(LS.llm, on ? '1' : '0');
        const hint = $('llm-mode-hint');
        if (hint) {
            hint.textContent = on ? 'LLM 参与开放式问题' : '纯本地规则，零 token';
            hint.className = 'tool-hint' + (on ? '' : ' off');
        }
    }

    function syncLlmAvailability(available) {
        llmAvailable = !!available;
        const box = $('llm-toggle');
        if (!box) return;
        if (!llmAvailable) {
            box.checked = false;
            box.disabled = true;
            const hint = $('llm-mode-hint');
            if (hint) hint.textContent = '未配置 Key，只能走规则引擎';
        } else {
            box.disabled = false;
        }
    }
    window.syncLlmAvailability = syncLlmAvailability;   // 供 index.html 里的健康检查回调复用

    // ================= 折叠与滚动 =================
    function reveal(id) {
        const el = $(id);
        if (!el) return;
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function addToTop() {
        if ($('to-top')) return;
        const btn = document.createElement('button');
        btn.id = 'to-top';
        btn.className = 'icon-btn';
        btn.textContent = '↑';
        btn.title = '回到顶部';
        btn.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
        document.body.appendChild(btn);
        window.addEventListener('scroll', () => {
            btn.classList.toggle('show', window.scrollY > 480);
        }, { passive: true });
    }

    // ================= 启动 =================
    function wireCharts() {
        // ECharts 是 canvas，主题与创建时机无关，只能在画完之后重涂一次。
        // 切 tab 与展开面板都会新建图表，所以在这两处补调用。
        $$('.tab').forEach(t => t.addEventListener('click', () => {
            setTimeout(() => retintCharts(document.documentElement.getAttribute('data-theme') === 'dark'), 900);
        }));
    }

    function init() {
        applyTheme(currentTheme());
        wireCharts();

        const saved = safeGet(LS.llm);
        setUseLlm(saved === null ? true : saved === '1', false);

        enhanceTables();
        addToTop();
        refreshStatus();
        setInterval(refreshStatus, 60000);

        // 内容是异步渲染的，用 MutationObserver 兜住"新表格出现时补样式"
        const pending = new Set();
        const obs = new MutationObserver(muts => {
            let hit = false;
            for (const m of muts) {
                for (const n of m.addedNodes) {
                    if (n.nodeType !== 1) continue;
                    if (n.matches && (n.matches('tbody, tr') || n.matches(TABLE_SELECTOR))) hit = true;
                    else if (n.querySelector && n.querySelector(TABLE_SELECTOR)) hit = true;
                    if (hit) break;
                }
                if (hit) break;
            }
            if (!hit) return;
            clearTimeout(pending.t);
            pending.t = setTimeout(() => enhanceTables(), 180);
        });
        obs.observe(document.body, { childList: true, subtree: true });

        document.addEventListener('keydown', e => {
            const tag = (e.target.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
            if (e.key === 't' || e.key === 'T') {
                API.toggleTheme();
            } else if (e.key === 'r' || e.key === 'R') {
                refreshStatus();
            }
        });

        if (window.matchMedia) {
            const mq = window.matchMedia('(prefers-color-scheme: dark)');
            const handler = e => { if (!safeGet(LS.theme)) applyTheme(e.matches ? 'dark' : 'light'); };
            mq.addEventListener ? mq.addEventListener('change', handler) : mq.addListener(handler);
        }
    }

    const API = {
        toggleTheme() {
            const next = (document.documentElement.getAttribute('data-theme') === 'dark') ? 'light' : 'dark';
            applyTheme(next);
            return next;
        },
        setUseLlm,
        useLlm,
        refreshStatus,
        refreshTheme() { retintCharts(document.documentElement.getAttribute('data-theme') === 'dark'); },
        enhanceTables,
        reveal,
        get llmAvailable() { return llmAvailable; }
    };
    window.UITweaks = API;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
