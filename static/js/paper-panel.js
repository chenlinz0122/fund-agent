/*
 * 模拟盘与自我训练面板
 *
 * 这里只负责"把接口结果如实画出来"。所有判断（闸门、冻结日、样本外天数、
 * 费用与仓位）都来自后端代码，前端不做任何计算或美化。
 * 特别是"样本外天数 = 0"这种情况，一定原样显示，不用绿色糊过去。
 */
(function () {
    'use strict';

    let currentAccount = null;
    let paChart = null;

    const $ = id => document.getElementById(id);
    const num = v => (v === null || v === undefined || isNaN(v) ? '-' : Number(v).toFixed(2));
    const pct = v => (v === null || v === undefined || isNaN(v) ? '-' : Number(v).toFixed(1) + '%');
    const updown = v => (v > 0 ? 'var(--color-up)' : (v < 0 ? 'var(--color-down)' : 'inherit'));

    function toast(msg) {
        if (typeof showToast === 'function') showToast(msg, 'error'); else alert(msg);
    }

    async function call(path, { method = 'GET', body = null } = {}, btn) {
        if (btn) btn.disabled = true;
        try {
            const resp = await fetch(path, {
                method,
                headers: body ? { 'Content-Type': 'application/json' } : undefined,
                body: body ? JSON.stringify(body) : undefined
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) { toast(data.detail || data.message || ('HTTP ' + resp.status)); return null; }
            return data;
        } catch (e) {
            toast('请求出错：' + e.message);
            return null;
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    function cards(el, items) {
        el.innerHTML = items.map(([label, val, color]) =>
            `<div class="metric-card"><div class="value" style="color:${color || 'inherit'}">${val}</div>` +
            `<div class="label">${label}</div></div>`).join('');
    }

    function verdict(level, html) {
        return `<div class="verdict ${level}">${html}</div>`;
    }

    // ================= 账户 =================
    window.openPaperAccount = async function (btn) {
        const codes = ($('pa-codes').value.match(/\d{6}/g) || []);
        const body = await call('/api/paper/accounts', {
            method: 'POST',
            body: { name: $('pa-name').value || '模拟盘', capital: Number($('pa-capital').value) || 100000,
                    codes, seed_buy: $('pa-seed').checked }
        }, btn);
        if (!body) return;
        toast('已开户，冻结日 ' + body.frozen_at);
        await loadAccounts();
        selectAccount(body.account.id);
    };

    async function loadAccounts() {
        const body = await call('/api/paper/accounts');
        if (!body) return;
        $('pa-list').innerHTML = body.accounts.length
            ? body.accounts.map(a => `
                <div class="pa-item ${a.id === currentAccount ? 'active' : ''}" onclick="selectAccount('${a.id}')">
                    <b>${a.name}</b> <span class="muted">${a.id}</span>
                    <span>权益 ${num(a.equity)}</span>
                    <span>持仓 ${a.positions}</span>
                    <span>待确认 ${a.pending}</span>
                    <span class="chip">${a.frozen_at ? '冻结 ' + a.frozen_at : '未冻结'}</span>
                </div>`).join('')
            : '<div class="empty-hint">还没有账户，上面开一个（初始资金只是数字，不花钱）</div>';
    }

    window.selectAccount = async function (id) {
        currentAccount = id;
        const detail = await call('/api/paper/accounts/' + id);
        if (!detail) return;
        $('pa-detail-card').style.display = 'block';
        $('pa-title').textContent = detail.account.name + ' · ' + detail.account.id;
        loadAccounts();
        renderDetail(detail);
    };

    function renderDetail(detail) {
        const s = detail.stats;
        const m = s.metrics || {};
        cards($('pa-metrics'), [
            ['当前权益', num(s.equity), 'inherit'],
            ['初始资金', num(s.initial_capital), 'inherit'],
            ['现金', num(s.cash), 'inherit'],
            ['持仓市值', num(s.positions_value), 'inherit'],
            ['累计收益', (m.total_return_pct ?? '-') + '%', updown(m.total_return_pct)],
            ['最大回撤', (m.max_drawdown_pct ?? '-') + '%', 'var(--color-down)'],
            ['夏普', num(m.sharpe), updown(m.sharpe)],
            ['已付费用', num(s.fees_paid), 'var(--color-down)'],
            ['费用占比', pct(s.fee_drag_pct), 'var(--color-down)'],
            ['已确认交易', s.confirmed_trades, 'inherit'],
            ['待确认订单', s.pending_orders, 'inherit'],
            ['被拒订单', s.rejected_orders, s.rejected_orders ? 'var(--color-down)' : 'inherit'],
        ]);

        const oos = s.out_of_sample || {};
        $('pa-oos').innerHTML = oos.evidence
            ? verdict('ok', `<b>样本外已有证据</b>：冻结日 ${oos.frozen_at} 之后观察 ${oos.days_observed} 个交易日，`
                + `账户口径收益 ${num(oos.return_pct)}%，确认成交 ${oos.confirmed_trades} 笔。<br>${oos.note || ''}`)
            : verdict('bad', `<b>还不构成验证</b>：${oos.note || '没有样本外数据'}。`
                + '下面的所有成绩都来自冻结日之前（样本内），只能说明规则跑得通，不能说明它有效。');

        $('pa-positions').innerHTML = detail.positions.length
            ? `<thead><tr><th>代码</th><th>买入日</th><th>份额</th><th>成本</th><th>净值</th>
                   <th>市值</th><th>浮动盈亏</th><th>持有天数</th><th>赎回费率</th><th>净赎回值</th></tr></thead>
               <tbody>${detail.positions.map(p => `<tr>
                 <td>${p.code}</td><td>${p.opened_date}</td><td>${num(p.shares)}</td>
                 <td>${num(p.unit_cost)}</td><td>${num(p.nav)}</td><td>${num(p.market_value)}</td>
                 <td style="color:${updown(p.pnl_pct)}">${num(p.pnl_pct)}%</td>
                 <td>${p.holding_days}</td>
                 <td>${p.penalty_applies ? '<span class="tag-chip bad">惩罚 ' + p.redemption_rate_pct + '%</span>'
                                          : p.redemption_rate_pct + '%'}</td>
                 <td>${num(p.liquidation_value)}</td></tr>`).join('')}</tbody>`
            : '<tbody><tr><td class="muted">空仓</td></tr></tbody>';

        $('pa-orders').innerHTML = detail.orders.length
            ? `<thead><tr><th>提交日</th><th>代码</th><th>方向</th><th>金额/份额</th>
                   <th>状态</th><th>确认日</th><th>成交净值</th><th>费用</th><th>说明</th></tr></thead>
               <tbody>${detail.orders.slice(0, 40).map(o => `<tr>
                 <td>${o.submitted_date}</td><td>${o.code}</td>
                 <td>${o.side === 'buy' ? '申购' : '赎回'}</td>
                 <td>${o.side === 'buy' ? num(o.amount) : num(o.shares)}</td>
                 <td>${statusText(o.status)}</td><td>${o.confirmed_date || '-'}</td>
                 <td>${o.nav ?? '-'}</td><td>${num(o.fee || 0)}</td>
                 <td class="muted">${(o.reason || o.note || '')}</td></tr>`).join('')}</tbody>`
            : '<tbody><tr><td class="muted">还没有订单</td></tr></tbody>';

        if (window.echarts && detail.account.equity_history.length > 1) {
            if (!paChart || paChart.isDisposed()) paChart = echarts.init($('pa-chart'));
            const pts = detail.account.equity_history;
            paChart.setOption({
                tooltip: { trigger: 'axis' },
                legend: { data: ['账户权益', '持仓市值', '现金'], top: 0 },
                grid: { left: 70, right: 24, top: 32, bottom: 40 },
                xAxis: { type: 'category', data: pts.map(p => p.date) },
                yAxis: { type: 'value', scale: true },
                series: [
                    { name: '账户权益', type: 'line', showSymbol: false, data: pts.map(p => p.equity),
                      lineStyle: { width: 2, color: '#1a73e8' } },
                    { name: '持仓市值', type: 'line', showSymbol: false, areaStyle: { opacity: .12 },
                      data: pts.map(p => p.holdings), lineStyle: { width: 1, color: '#ea4335' } },
                    { name: '现金', type: 'line', showSymbol: false, data: pts.map(p => p.cash),
                      lineStyle: { width: 1, type: 'dashed', color: '#9aa0a6' } }
                ]
            }, true);
        }
        if (window.UITweaks) { UITweaks.enhanceTables($('pa-detail-card')); UITweaks.refreshTheme(); }
    }

    function statusText(s) {
        return { pending: '待确认', confirmed: '已成交', rejected: '已拒单' }[s] || s;
    }

    window.settleAccount = async function () {
        if (!currentAccount) return toast('请先选择账户');
        const body = await call(`/api/paper/accounts/${currentAccount}/settle`, { method: 'POST' });
        if (!body) return;
        toast(body.confirmed.length ? `已确认 ${body.confirmed.length} 笔` : '没有可确认的订单（T+1 才有净值）');
        selectAccount(currentAccount);
    };

    window.autoStepAccount = async function (btn) {
        if (!currentAccount) return toast('请先选择账户');
        const body = await call(`/api/paper/accounts/${currentAccount}/auto`, { method: 'POST' }, btn);
        if (!body) return;
        renderDecision(body);
        toast(`结算 ${body.settled.length} 笔，新下单 ${body.submitted.length} 笔`);
        selectAccount(currentAccount);
    };

    window.previewDecision = async function (btn) {
        if (!currentAccount) return toast('请先选择账户');
        const body = await call(`/api/paper/accounts/${currentAccount}/decision`, {}, btn);
        if (body) renderDecision({ plan: body });
    };

    function renderDecision(payload) {
        const plan = payload.plan || {};
        const gates = (plan.gates || []).map(g =>
            `<span class="tag-chip">${g.name} ${Math.round(g.factor * 100)}%</span>`).join('');
        const orders = (payload.submitted || plan.orders || []);
        $('pa-decision').innerHTML = `
            <div class="verdict ${plan.target_exposure !== undefined ? 'ok' : 'warn'}">
                <b>今日决策</b>：当前敞口 ${pct(plan.current_exposure !== undefined ? plan.current_exposure * 100 : null)}
                → 目标 ${pct(plan.target_exposure !== undefined ? plan.target_exposure * 100 : null)}
                （实际执行 ${pct(plan.applied_exposure !== undefined ? plan.applied_exposure * 100 : null)}）
                ${gates}
                <div class="muted">${(plan.state ? Object.entries(plan.state).map(([k, v]) => `${k}=${v}`).join('　') : '')}</div>
                ${orders.length ? '<div>本次动作：' + orders.map(o =>
                    `${o.side === 'buy' ? '申购 ' + o.code + ' ' + num(o.amount) + ' 元' : '赎回 ' + o.code + ' ' + num(o.shares) + ' 份'}
                     [${statusText(o.status)}${o.note ? '·' + o.note : ''}]`).join('；') + '</div>'
                  : '<div>本次不动手（在不交易带内，或规则认为无需调整）</div>'}
                ${(payload.equity && payload.equity.out_of_sample && !payload.equity.out_of_sample.evidence)
                    ? '<div class="muted">注意：' + payload.equity.out_of_sample.note + '</div>' : ''}
            </div>`;
    }

    window.reviewAccount = async function (btn) {
        if (!currentAccount) return toast('请先选择账户');
        const body = await call(`/api/train/review?account_id=${currentAccount}`, {}, btn);
        if (!body) return;
        const facts = Object.entries(body.facts || {}).map(([k, v]) =>
            `<tr><td>${k}</td><td class="num">${typeof v === 'object' ? JSON.stringify(v) : (v ?? '-')}</td></tr>`).join('');
        $('pa-review').innerHTML = `
            <h4>复盘报告（数字由代码算，叙述由 ${body.narration.engine === 'llm' ? 'LLM 写' : '模板写'}）</h4>
            ${body.evidence ? verdict('ok', '样本外已有数据，见下表') : verdict('bad', body.note || '样本外天数为 0，当前成绩不构成验证')}
            <div class="two-col">
              <div class="table-wrapper"><table class="data-table"><tbody>${facts}</tbody></table></div>
              <div class="narrative">${(body.narration.text || '').replace(/\n/g, '<br>')}</div>
            </div>`;
        if (window.UITweaks) UITweaks.enhanceTables($('pa-review'));
    };

    // ================= 自训练 =================
    window.runTrainRound = async function (btn) {
        const codes = ($('tr-codes').value.match(/\d{6}/g) || []);
        if (codes.length < 2) return toast('请至少输入 2 只基金代码');
        toast('进化中：会评估几十到几百组参数，请稍候');
        const body = await call('/api/train/round', {
            method: 'POST',
            body: { codes, days: Number($('tr-days').value) || 1200,
                    pop_size: Number($('tr-pop').value) || 12,
                    generations: Number($('tr-gen').value) || 3,
                    holdout_days: Number($('tr-holdout').value) || 60 }
        }, btn);
        if (!body) return;
        const g = body.gates, h = body.holdout;
        $('tr-result').innerHTML = `
            ${body.promotable
                ? verdict('ok', `<b>三重闸门全过</b>，可以晋升。${body.promotion_note}`)
                : verdict('bad', `<b>闸门未过，不晋升。</b>${body.promotion_note}`)}
            <div class="metrics-grid">
                <div class="metric-card"><div class="value">${body.evolution.evaluations}</div><div class="label">评估参数组数</div></div>
                <div class="metric-card"><div class="value">${num(body.record.score)}</div><div class="label">挑战者得分</div></div>
                <div class="metric-card"><div class="value" style="color:${g.plateau_ok ? 'var(--color-up)' : 'var(--color-down)'}">${g.plateau_ok ? '过' : '不过'}</div><div class="label">参数平原</div></div>
                <div class="metric-card"><div class="value" style="color:${g.dsr_ok ? 'var(--color-up)' : 'var(--color-down)'}">${g.dsr_ok ? '过' : '不过'}</div><div class="label">Deflated Sharpe</div></div>
                <div class="metric-card"><div class="value" style="color:${g.walk_forward_ok ? 'var(--color-up)' : 'var(--color-down)'}">${g.walk_forward_ok ? '过' : '不过'}</div><div class="label">walk-forward</div></div>
                <div class="metric-card"><div class="value">${num(h.total_return_pct)}%</div><div class="label">留出段收益</div></div>
                <div class="metric-card"><div class="value">${num(h.alpha_vs_dca_pct)}%</div><div class="label">留出段超额(vs定投)</div></div>
                <div class="metric-card"><div class="value">${num(h.max_drawdown_pct)}%</div><div class="label">留出段最大回撤</div></div>
            </div>
            <div class="muted">训练段 ${body.split.train_days} 天 / 留出段 ${body.split.holdout_days} 天
              (${(body.split.holdout_range || []).join(' ~ ')})；留出段在进化时被整段藏起来，这是该策略第一次见。</div>
            ${(g.failures || []).map(f => verdict('warn', f)).join('')}`;
        loadArchive();
    };

    async function loadArchive() {
        const body = await call('/api/train/archive');
        if (!body) return;
        const s = body.summary;
        $('ar-summary').innerHTML = `
            <div class="chips-row">
                <span class="chip ok"><span class="dot"></span>冠军 ${s.champions}</span>
                <span class="chip"><span class="dot"></span>挑战者 ${s.challengers}</span>
                <span class="chip err"><span class="dot"></span>被拒 ${s.rejected}</span>
                <span class="chip"><span class="dot"></span>退役 ${s.retired}</span>
            </div><div class="muted">${body.rule}</div>`;
        $('ar-records').innerHTML = `<thead><tr><th>ID</th><th>创建</th><th>状态</th><th>得分</th>
                <th>尝试次数</th><th>平原</th><th>DSR</th><th>WF</th><th>冻结日</th><th>操作</th></tr></thead>
            <tbody>${body.records.map(r => `<tr>
                <td>${r.id}</td><td>${r.created_at}</td>
                <td>${statusName(r.status)}</td><td>${num(r.score)}</td><td>${r.n_trials}</td>
                <td>${r.gates.plateau_ok ? '✅' : '❌'}</td>
                <td>${r.gates.dsr_ok ? '✅' : '❌'}</td>
                <td>${r.gates.walk_forward_ok ? '✅' : '❌'}</td>
                <td>${r.frozen_at || '-'}</td>
                <td>${r.gates.passed && r.status !== 'champion'
                    ? `<button class="btn btn-secondary btn-sm" onclick="promoteRecord('${r.id}', this)">晋升为冠军</button>`
                    : (r.status === 'champion' ? '<span class="muted">当前冠军</span>' : '<span class="muted">闸门未过</span>')}</td>
            </tr>`).join('') || '<tr><td colspan="10" class="muted">还没有记录，先跑一轮</td></tr>'}</tbody>`;
        if (window.UITweaks) UITweaks.enhanceTables($('ar-records'));
    }

    function statusName(s) {
        return { champion: '冠军', challenger: '挑战者', rejected: '被拒', retired: '退役' }[s] || s;
    }

    window.promoteRecord = async function (id, btn) {
        if (!confirm('晋升为冠军会把今天记为「冻结日」，此后净值才算样本外证据。确认？')) return;
        const body = await call('/api/train/promote', { method: 'POST', body: { record_id: id } }, btn);
        if (!body) return;
        toast('已晋升，冻结日 ' + body.frozen_at);
        loadArchive();
    };

    // ================= 挂载 =================
    function init() {
        if (!$('pa-list')) return;
        loadAccounts();
        loadArchive();
        document.querySelectorAll('.tab').forEach(t => {
            t.addEventListener('click', () => {
                if (t.dataset.view === 'paper') { loadAccounts(); loadArchive(); }
            });
        });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
