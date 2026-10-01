/*
 * 期望值 / 稳健性 / 策略模拟面板
 *
 * 单独成文件的原因（见 ARCHITECTURE.md 第 5 条）：index.html 已经过大，
 * 新功能不再往里堆。这里只负责"调接口 + 画结果"，不含任何计算逻辑——
 * 所有判断都在后端 analysis/ 里，前端只如实转述。
 */
(function () {
    'use strict';

    let simChart = null;

    function codes() {
        const raw = (document.getElementById('portfolio-codes') || {}).value || '';
        return (raw.match(/\d{6}/g) || []).filter((v, i, a) => a.indexOf(v) === i);
    }

    function days() {
        const sel = document.getElementById('portfolio-days');
        return sel ? Number(sel.value) || 365 : 365;
    }

    async function post(path, body, btn) {
        if (codes().length < 2) { toast('请至少输入 2 只 6 位基金代码'); return null; }
        if (btn) btn.disabled = true;
        try {
            const resp = await fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(Object.assign({ codes: codes(), days: days() }, body || {}))
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) { toast((data.detail || data.message || '请求失败') + ' (' + resp.status + ')'); return null; }
            return data;
        } catch (e) {
            toast('请求出错：' + e.message);
            return null;
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    async function get(path, btn) {
        if (codes().length < 1) { toast('请先输入基金代码'); return null; }
        if (btn) btn.disabled = true;
        try {
            const resp = await fetch(path);
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) { toast((data.detail || data.message || '请求失败') + ' (' + resp.status + ')'); return null; }
            return data;
        } catch (e) {
            toast('请求出错：' + e.message);
            return null;
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    function toast(msg) {
        if (typeof showToast === 'function') { showToast(msg, 'error'); return; }
        alert(msg);
    }

    function show(id) {
        const el = document.getElementById(id);
        if (el) el.style.display = 'block';
        // 新渲染的表格要补上数值对齐/涨跌着色，并平滑滚到可视区
        if (window.UITweaks) {
            try { UITweaks.enhanceTables(el); } catch (e) { /* 增强失败不影响内容 */ }
            setTimeout(() => { UITweaks.reveal(id); UITweaks.refreshTheme(); }, 120);
        }
        return el;
    }

    /** 结论条：ok / warn / bad 三档，避免"看起来都差不多"的一堆文字 */
    function verdict(level, html) {
        const cls = level === 'ok' ? 'ok' : (level === 'bad' ? 'bad' : 'warn');
        return `<div class="verdict ${cls}">${html}</div>`;
    }

    const updown = v => (v > 0 ? 'var(--color-up)' : (v < 0 ? 'var(--color-down)' : 'inherit'));
    const num = v => (v === null || v === undefined || isNaN(v) ? '-' : Number(v).toFixed(2));
    const pct = v => (v === null || v === undefined || isNaN(v) ? '-' : Number(v).toFixed(1) + '%');

    function cards(el, items) {
        el.innerHTML = items.map(([label, val, color]) =>
            `<div class="metric-card"><div class="value" style="color:${color || 'inherit'}">${val}</div>` +
            `<div class="label">${label}</div></div>`).join('');
    }

    // ========== 1. 费后期望对照 ==========
    window.runExpectancy = async function (btn) {
        const body = await post('/api/portfolio/expectancy', { round_trips_per_year: 4 }, btn);
        if (!body) return;
        const el = show('sim-expectancy');
        const rows = (body.rows || []).map(r => r.error
            ? `<tr><td>${r.code}</td><td colspan="6">数据不足</td></tr>`
            : `<tr>
                 <td>${r.code}</td>
                 <td style="color:${updown(r.gross_annual_return_pct)}">${num(r.gross_annual_return_pct)}%</td>
                 <td>-${num(r.annualized_fee_drag_pct)}%</td>
                 <td style="color:${updown(r.net_annual_return_pct)}">${num(r.net_annual_return_pct)}%</td>
                 <td style="color:${updown(r.net_sharpe)}">${num(r.net_sharpe)}</td>
                 <td>${r.positive_expectancy ? '✅ 正' : '⚠️ 非正'}</td>
                 <td>${(r.risk_points || []).join('、') || '无'}</td>
               </tr>`).join('');
        el.querySelector('#exp-table').innerHTML = `
            <thead><tr><th>代码</th><th>毛年化</th><th>费用侵蚀/年</th><th>费后年化</th>
                     <th>净夏普</th><th>期望方向</th><th>命中的风险点</th></tr></thead>
            <tbody>${rows}</tbody>`;
        el.querySelector('#exp-note').textContent = body.assumption + '（本次按每年往返 ' + body.round_trips_per_year + ' 次估算）';
    };

    // ========== 2. 稳健性体检 ==========
    window.runRobustness = async function (btn) {
        const body = await post('/api/portfolio/robustness', { folds: 4, train_days: 250, test_days: 60, n_trials: 10 }, btn);
        if (!body) return;
        const el = show('sim-robust');
        const wf = body.walk_forward || {};
        const verdictEl = el.querySelector('#robust-verdict');

        if (!wf.available) {
            verdictEl.innerHTML = verdict('warn', `<b>样本外检验无法进行</b>：${wf.note || '历史不足'}`);
        } else {
            const ok = wf.effective_out_of_sample;
            verdictEl.innerHTML = verdict(ok ? 'ok' : 'bad', `
                <b>${ok ? '排名在样本外仍有预测力' : '排名在样本外接近随机'}</b><br>
                ${wf.folds} 折滚动检验，平均 Rank IC = <b>${num(wf.mean_rank_ic)}</b>
                （标准差 ${num(wf.ic_std)}，t = ${wf.ic_t_stat ?? '-'}），
                头部命中率 ${pct(wf.mean_hit_rate_pct)}。<br>
                ${wf.interpretation || ''}`);
        }

        const dsr = (body.per_fund_deflated_sharpe || []).map(f => `
            <tr>
              <td>${f.code}</td>
              <td>${num(f.sharpe)}</td>
              <td>${num(f.deflated_sharpe)}</td>
              <td>${f.robust_at_95 ? '✅ 稳' : '❌ 不稳'}</td>
              <td style="font-size:12px;color:var(--text-secondary)">${f.interpretation || ''}</td>
            </tr>`).join('');
        el.querySelector('#robust-table').innerHTML = `
            <thead><tr><th>代码</th><th>夏普</th><th>Deflated Sharpe</th><th>95% 稳健</th><th>结论</th></tr></thead>
            <tbody>${dsr}</tbody>`;
    };

    // ========== 3. 当前仓位档位 ==========
    window.runPolicy = async function (btn) {
        const url = `/api/portfolio/policy?codes=${codes().join(',')}&days=${days()}`;
        const body = await get(url, btn);
        if (!body) return;
        const el = show('sim-policy');
        el.querySelector('#policy-rows').innerHTML = (body.rows || []).map(r => {
            if (!r.available) return `<div class="policy-row"><b>${r.code}</b>：${r.note}</div>`;
            const t = r.target_exposure;
            const e = Math.round(t.exposure * 100);
            const bars = (t.gates || []).map(g =>
                `<span class="gate-chip">${g.name} ${Math.round(g.factor * 100)}%</span>`).join('');
            return `<div class="policy-row">
                <b>${r.code}</b> 建议敞口 <b style="color:${e >= 80 ? 'var(--color-up)' : (e <= 40 ? 'var(--color-down)' : 'inherit')}">${e}%</b>
                ${bars}
                <div class="policy-reason">${(t.reasons || []).join('；')}</div>
            </div>`;
        }).join('');
        el.querySelector('#policy-disclaimer').textContent = body.disclaimer;
    };

    // ========== 4. 策略模拟 ==========
    window.runSimulate = async function (btn) {
        const body = await post('/api/portfolio/simulate', { season: '夏' }, btn);
        if (!body) return;
        const el = show('sim-strategy');
        const res = body.result || {};
        const st = res.strategy || {};
        const m = st.metrics || {};

        cards(el.querySelector('#sim-metrics'), [
            ['策略累计', num(m.total_return_pct) + '%', updown(m.total_return_pct)],
            ['定投基准', num(st.baseline_dca_pct) + '%', updown(st.baseline_dca_pct)],
            ['超额(vs 定投)', num(res.alpha_vs_dca_pct) + '%', updown(res.alpha_vs_dca_pct)],
            ['最大回撤', num(m.max_drawdown_pct) + '%', 'var(--color-down)'],
            ['夏普', num(m.sharpe), updown(m.sharpe)],
            ['平均敞口', pct(res.avg_exposure_pct), 'inherit'],
            ['换手合计', num(st.total_turnover_pct) + '%', 'inherit'],
            ['费用拖累', '-' + num(st.total_cost_pct) + '%', 'var(--color-down)'],
            ['调仓次数', st.trade_count ?? '-', 'inherit'],
            ['熔断触发', res.fused ? '⚠️ 是' : '否', res.fused ? 'var(--color-down)' : 'inherit'],
        ]);

        const notes = [];
        (body.warnings || []).forEach(w => notes.push(['warn', w]));
        const guardNotes = (res.guard && res.guard.notes) || [];
        // guard.notes 默认是"约束声明"（中性偏正常），只有真的越界才标红
        guardNotes.forEach(n => notes.push([/不达标|超过上限|违反|未满足/.test(n) ? 'bad' : 'ok', n]));
        if (body.ignored_reason) notes.push(['warn', body.ignored_reason]);
        el.querySelector('#sim-warn').innerHTML = notes.map(([lv, txt]) => verdict(lv, txt)).join('');

        const dom = el.querySelector('#sim-chart');
        if (window.echarts) {
            if (!simChart || simChart.isDisposed()) simChart = echarts.init(dom);
            const curve = (st.equity_curve || []).map(p => [p.date, p.nav]);
            const expo = (st.exposure_curve || []).map(p => [p.date, p.exposure]);
            // 回撤"水下曲线"：比最大回撤一个数字直观得多（借自 akquant-lwc 报告分区）
            let peak = -Infinity;
            const underwater = (st.equity_curve || []).map(p => {
                peak = Math.max(peak, p.nav);
                return [p.date, Math.round((p.nav / peak - 1) * 10000) / 100];
            });
            simChart.setOption({
                tooltip: { trigger: 'axis' },
                axisPointer: { link: [{ xAxisIndex: 'all' }] },
                legend: { data: ['策略净值', '目标敞口%', '回撤'], top: 0 },
                grid: [
                    { left: 55, right: 55, top: 34, height: '48%' },
                    { left: 55, right: 55, top: '68%', height: '20%' }
                ],
                xAxis: [
                    { type: 'time', gridIndex: 0, axisLabel: { show: false } },
                    { type: 'time', gridIndex: 1 }
                ],
                yAxis: [
                    { type: 'value', gridIndex: 0, name: '净值', scale: true },
                    { type: 'value', gridIndex: 0, name: '敞口%', min: 0, max: 100, position: 'right', splitLine: { show: false } },
                    { type: 'value', gridIndex: 1, name: '回撤%', max: 0, splitNumber: 2 }
                ],
                dataZoom: [{ type: 'inside', xAxisIndex: [0, 1] }, { type: 'slider', xAxisIndex: [0, 1], height: 16, bottom: 6 }],
                series: [
                    { name: '策略净值', type: 'line', xAxisIndex: 0, yAxisIndex: 0, showSymbol: false, data: curve,
                      lineStyle: { width: 2, color: '#1a73e8' },
                      areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                          colorStops: [{ offset: 0, color: 'rgba(26,115,232,.22)' }, { offset: 1, color: 'rgba(26,115,232,0)' }] } } },
                    { name: '目标敞口%', type: 'line', xAxisIndex: 0, yAxisIndex: 1, step: 'end', showSymbol: false,
                      data: expo, lineStyle: { width: 1, color: '#ea4335', type: 'dashed' } },
                    { name: '回撤', type: 'line', xAxisIndex: 1, yAxisIndex: 2, showSymbol: false, data: underwater,
                      lineStyle: { width: 1, color: '#34a853' },
                      areaStyle: { color: 'rgba(52,168,83,.25)' } }
                ]
            }, true);
        }

        el.querySelector('#sim-trades').innerHTML = (st.trades || []).slice(-15).map(t =>
            `<tr><td>${t.date}</td><td>${t.action}</td><td>${t.from}% → ${t.to}%</td>
             <td style="font-size:12px;color:var(--text-secondary)">${t.reason || ''}</td></tr>`).join('')
            || '<tr><td colspan="4">区间内没有触发过调仓（不交易带吃掉了所有小幅偏差）</td></tr>';
    };

    // ========== 5. 参数进化 ==========
    window.runEvolve = async function (btn) {
        toast('进化中：会评估几十到几百组参数，请稍候');
        const body = await post('/api/portfolio/evolve', { pop_size: 12, generations: 3, plateau: true }, btn);
        if (!body) return;
        const el = show('sim-evolve');
        const ch = body.challenger || {};
        const oc = body.overfitting_check || {};
        const pl = body.plateau || {};
        const review = body.final_review || {};

        cards(el.querySelector('#evolve-metrics'), [
            ['评估参数组数', body.evaluations ?? '-', 'inherit'],
            ['挑战者得分', num(ch.score), updown(ch.score)],
            ['Deflated Sharpe', num(oc.deflated_sharpe), oc.robust_at_95 ? 'var(--color-up)' : 'var(--color-down)'],
            ['95% 稳健', oc.robust_at_95 ? '✅ 是' : '❌ 否', oc.robust_at_95 ? 'var(--color-up)' : 'var(--color-down)'],
            ['参数平原', pl.on_plateau ? '✅ 在平原' : '⚠️ 在针尖', pl.on_plateau ? 'var(--color-up)' : 'var(--color-down)'],
            ['资金耗尽概率', pct((review.ruin_probability || 0) * 100), (review.ruin_probability || 0) > 0.2 ? 'var(--color-down)' : 'inherit'],
            ['中位重抽样收益', num(review.median_return_pct) + '%', updown(review.median_return_pct)],
            ['自动晋升', '否', 'inherit'],
        ]);

        const ocLevel = oc.robust_at_95 ? 'ok' : 'bad';
        el.querySelector('#evolve-verdict').innerHTML =
            verdict(pl.on_plateau ? 'ok' : 'bad', `<b>参数平原：</b>${pl.interpretation || '未完成平原检验'}`) +
            verdict(ocLevel, `<b>Deflated Sharpe：</b>${(oc.deflated_sharpe ?? '-')} —— ${oc.interpretation || ''}`) +
            verdict('warn', `<b>为什么不会自动采用：</b>${(body.promotion || {}).note || ''}`) +
            `<div style="font-size:12px;color:var(--text-secondary);margin-top:6px;">${body.cost_note || ''}</div>`;

        el.querySelector('#evolve-top').innerHTML = `
            <thead><tr><th>#</th><th>得分</th><th>超额(vs定投)</th><th>费用拖累</th><th>调仓次数</th><th>来源</th></tr></thead>
            <tbody>${(body.top || []).map((r, i) => `<tr>
                <td>${i + 1}</td><td>${num(r.score)}</td><td>${num(r.alpha)}%</td>
                <td>-${num(r.cost)}%</td><td>${r.trades}</td><td>${originName(r.origin)}</td></tr>`).join('')}</tbody>`;

        el.querySelector('#evolve-genes').innerHTML = `<code>${JSON.stringify((ch.genes) || {}, null, 1)}</code>`;
        el.querySelector('#evolve-env').textContent = (body.environment || {}).note || '';
    };

    function originName(o) {
        return { incumbent: '旧冠军', mutant: '精英变异', explorer: '随机移民', offspring: '交叉子代' }[o] || o;
    }
})();
