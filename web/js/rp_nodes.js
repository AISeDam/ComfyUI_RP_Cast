/**
 * ComfyUI_RP_Cast - Frontend Extension v0.5.40
 */

// ComfyUI 0.17+ Vite build: app is on window but may not be ready yet.
// Poll until app is available, then register.
(function bootstrap() {
    const app = window.app;
    if (app && typeof app.registerExtension === 'function') {
        _registerRPCast(app);
    } else {
        // Retry up to 50 times (5 seconds)
        let tries = 0;
        const timer = setInterval(() => {
            const a = window.app;
            if (a && typeof a.registerExtension === 'function') {
                clearInterval(timer);
                _registerRPCast(a);
            } else if (++tries > 50) {
                clearInterval(timer);
                console.error('[RP_Cast] app.registerExtension not found after 5s');
            }
        }, 100);
    }
})();

function _registerRPCast(app) {


const RP_COLORS = [
    "#4A90D9","#E8A838","#7DC97D","#C97D7D",
    "#9B7DC9","#7DC9C9","#D97D4A","#D94A90","#90D94A","#4AD9C9",
];

const CANVAS_H = 100;

// ── ratio 파싱 ──────────────────────────────────────────
function parseRatio2D(str) {
    if (!str || str.trim() === "auto") return [[1]];
    const rows = String(str).split(";").map(row =>
        row.split(",").map(s => { const v=parseFloat(s.trim()); return(isNaN(v)||v<=0)?1:v; })
    ).filter(r=>r.length>0);
    return rows.length ? rows : [[1]];
}

// ── 프롬프트 → ratioStr ─────────────────────────────────
// README 2D region assignment 기준
// Columns(Horizontal): ADDROW→행(;), ADDCOL→열(,), 세그먼트=구분자+1
// Rows(Vertical):      ADDCOL→행(;), ADDROW→열(,)
function promptToRatio(prompt, divideMode) {
    let p = String(prompt||"");
    // ADDCOMM, ADDBASE 이후 텍스트로 파싱 (BASE 세그먼트 포함)
    if (p.includes("ADDCOMM")) p = p.split("ADDCOMM")[1] || "";
    if (p.includes("ADDBASE")) p = p.split("ADDBASE")[1] || "";

    const hasCol = p.includes("ADDCOL");
    const hasRow = p.includes("ADDROW");
    const isVertical = (divideMode||"Horizontal").includes("Ver");
    const countOf = (str, kw) => (str.split(kw).length - 1);

    if (!hasCol && !hasRow) return "1";

    if (isVertical) {
        // Vertical(Rows): ADDCOL→행(;), ADDROW→열(,)
        if (hasCol && hasRow) {
            return p.split("ADDCOL")
                .map(seg => Array(countOf(seg, "ADDROW") + 1).fill("1").join(","))
                .join(";");
        } else if (hasCol) {
            return Array(countOf(p, "ADDCOL") + 1).fill("1").join(";");
        } else {
            return Array(countOf(p, "ADDROW") + 1).fill("1").join(",");
        }
    } else {
        // Horizontal(Columns): ADDROW→행(;), ADDCOL→열(,)
        if (hasCol && hasRow) {
            return p.split("ADDROW")
                .map(seg => Array(countOf(seg, "ADDCOL") + 1).fill("1").join(","))
                .join(";");
        } else if (hasRow) {
            return Array(countOf(p, "ADDROW") + 1).fill("1").join(";");
        } else {
            return Array(countOf(p, "ADDCOL") + 1).fill("1").join(",");
        }
    }
}

// ── 링크를 통해 upstream 위젯 값 읽기 ───────────────────
// forceInput 위젯은 w.value가 갱신되지 않음
// 연결된 upstream 노드의 출력 위젯 값을 직접 읽어야 함
function getLinkedValue(node, inputName) {
    try {
        const inp = node.inputs?.find(i => i.name === inputName);
        if (!inp?.link) return null;

        const link = node.graph.links[inp.link];
        if (!link) return null;

        const srcNode = node.graph._nodes_by_id[link.origin_id];
        if (!srcNode) return null;

        const slotIdx = link.origin_slot;

        // 방법1: output에 해당하는 위젯 값 (위젯이 output으로 연결된 경우)
        if (srcNode.widgets_values && slotIdx < srcNode.widgets_values.length) {
            // 출력 슬롯이 위젯과 매핑된 경우 시도
        }

        // 방법2: RETURN_NAMES 기준으로 output 슬롯 → 위젯 이름 매핑
        // RPPromptParser의 outputs: [regional_prompts_nolora, regional_lora_map, divide_ratio]
        // divide_ratio는 index 2
        const outputNames = srcNode.outputs?.map(o => o.name) || [];
        const outName = outputNames[slotIdx];

        // 소스 노드의 위젯에서 같은 이름의 값 찾기
        const srcWidget = srcNode.widgets?.find(w =>
            w.name === outName || w.name === "_divide_ratio"
        );
        if (srcWidget?.value) return srcWidget.value;

        // 방법3: widgets_values 배열에서 슬롯 인덱스 기준으로 읽기
        if (srcNode.widgets_values) {
            // non-forceInput 위젯들의 값 순서로 접근
            const nonForceWidgets = srcNode.widgets?.filter(w =>
                !w.options?.forceInput
            ) || [];
            const wIdx = nonForceWidgets.findIndex(w =>
                w.name === "divide_ratio" || w.name === outName
            );
            if (wIdx >= 0 && srcNode.widgets_values[wIdx] !== undefined) {
                return String(srcNode.widgets_values[wIdx]);
            }
        }

        return null;
    } catch(e) {
        return null;
    }
}

// ── 시각화 그리기 ────────────────────────────────────────
function drawVisualize(ctx, ratioStr, divMode, x, y, w, h) {
    const PANEL_H = h - 22;
    const COORD_H = 18;
    const r2d  = parseRatio2D(ratioStr);
    const is2D = r2d.length > 1 || String(ratioStr).includes(";");
    const isH  = (divMode||"Horizontal")==="Horizontal";

    ctx.fillStyle="#1e1e1e"; ctx.fillRect(x,y,w,PANEL_H);
    ctx.strokeStyle="#555"; ctx.lineWidth=1; ctx.strokeRect(x,y,w,PANEL_H);

    let ci=0;
    if (is2D) {
        const nR=r2d.length;
        r2d.forEach((cols,ri)=>{
            const ct=cols.reduce((a,b)=>a+b,0)||1; let cc=0;
            cols.forEach((cr,cj)=>{
                const cf=cr/ct, rf=1/nR, color=RP_COLORS[ci++%RP_COLORS.length];
                const [rx,ry,rw,rh]=isH?[x+cc*w,y+ri*rf*PANEL_H,cf*w,rf*PANEL_H]:[x+ri*rf*w,y+cc*PANEL_H,rf*w,cf*PANEL_H];
                ctx.fillStyle=color+"99"; ctx.fillRect(rx,ry,rw,rh);
                ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.strokeRect(rx,ry,rw,rh);
                ctx.fillStyle="#fff"; ctx.textAlign="center"; ctx.textBaseline="middle";
                ctx.font="bold 10px Arial"; ctx.fillText(isH?`[${ri},${cj}]`:`[${cj},${ri}]`,rx+rw/2,ry+rh/2-5);
                ctx.font="9px Arial"; ctx.fillText(`${Math.round(rf*cf*100)}%`,rx+rw/2,ry+rh/2+7);
                cc+=cf;
            });
        });
    } else {
        const parts=r2d[0]||[1],tot=parts.reduce((a,b)=>a+b,0)||1; let c=0;
        parts.forEach((p,i)=>{
            const f=p/tot,color=RP_COLORS[i%RP_COLORS.length];
            const [rx,ry,rw,rh]=isH?[x+c*w,y,f*w,PANEL_H]:[x,y+c*PANEL_H,w,f*PANEL_H];
            ctx.fillStyle=color+"99"; ctx.fillRect(rx,ry,rw,rh);
            ctx.strokeStyle=color; ctx.lineWidth=1; ctx.strokeRect(rx,ry,rw,rh);
            ctx.fillStyle="#fff"; ctx.textAlign="center"; ctx.textBaseline="middle";
            ctx.font="bold 10px Arial"; ctx.fillText(`div${i}`,rx+rw/2,ry+rh/2-5);
            ctx.font="9px Arial"; ctx.fillText(`${c.toFixed(2)}~${(c+f).toFixed(2)}`,rx+rw/2,ry+rh/2+7);
            c+=f;
        });
    }

    const ty=y+PANEL_H+2;
    ctx.fillStyle="#1a1a2e"; ctx.fillRect(x,ty,w,COORD_H);
    ctx.strokeStyle="#333"; ctx.strokeRect(x,ty,w,COORD_H);
    ctx.fillStyle="#8888cc"; ctx.font="10px monospace";
    ctx.textAlign="left"; ctx.textBaseline="middle";
    let coord;
    if (is2D) {
        coord=r2d.map((cols,ri)=>{
            const t=cols.reduce((a,b)=>a+b,0)||1; let c=0;
            const prefix=isH?`R${ri}`:`C${ri}`;
            return `${prefix}:[${cols.map(v=>{const s=(c/t).toFixed(2);c+=v;return`${s}~${(c/t).toFixed(2)}`;}).join(",")}]`;
        }).join(" ");
    } else {
        const pts=r2d[0]||[1],t=pts.reduce((a,b)=>a+b,0)||1; let c=0;
        coord=(isH?"H":"V")+": "+pts.map(p=>{const s=(c/t).toFixed(2);c+=p;return`[${s}~${(c/t).toFixed(2)}]`;}).join(" ");
    }
    ctx.fillText(coord, x+4, ty+COORD_H/2);
}

// ── customCanvas 위젯 추가 ───────────────────────────────
function addRatioCanvas(node, getRatio, getDivMode) {
    // Prevent duplicate canvas widgets
    if (node.widgets?.some(w => w.name === "rp-ratio-canvas")) return;
    const widget = {
        type: "customCanvas",
        name: "rp-ratio-canvas",
        draw(ctx, node, widgetWidth, widgetY) {
            const ratioStr = getRatio() || "1,1";
            const divMode  = getDivMode ? getDivMode() : "Horizontal";
            drawVisualize(ctx, ratioStr, divMode, 4, widgetY, widgetWidth-8, CANVAS_H);
        },
        computeSize(width) { return [width||0, CANVAS_H]; },
    };
    node.addCustomWidget(widget);
    node.onResize = function(size) {};
    return widget;
}


// ── RPPromptParser 초기화 함수 ───────────────────────────────────────────────
// onNodeCreated 및 setup() 양쪽에서 호출. 중복 실행 방지 포함.
function _initPP(node) {
    // Already initialized if canvas widget exists
    if (node.widgets?.some(w => w.name === 'rp-ratio-canvas')) return;

    // Add buttons (guard by checking existing button names)
    if (!node.widgets?.some(w => w.name === '+ ADDCOMM')) {
        for (const kw of ["ADDCOMM","ADDBASE","ADDCOL","ADDROW"]) {
            node.addWidget("button", `+ ${kw}`, null, () => {
                const pw = node.widgets.find(w => w.name === "prompt");
                if (pw) {
                    pw.value = (pw.value||"").trimEnd() + `\n${kw}\n`;
                    node._rpAutoUpdate();
                }
            });
        }
        node.addWidget("button", "▶ Visualize Regions", null, () => {
            node.setDirtyCanvas(true);
        });
    }

    // 캔버스 위젯 추가
    const _getRatio = () => {
        const dw = node.widgets?.find(w => w.name === "divide_ratio");
        const mw = node.widgets?.find(w => w.name === "divide_mode");
        const pw = node.widgets?.find(w => w.name === "prompt");
        const aw = node.widgets?.find(w => w.name === "auto_div_calc");
        if (aw?.value === "manual") {
            const v = (dw?.value || "").trim();
            return (v && v.toLowerCase() !== "auto") ? v
                : promptToRatio(pw?.value||"", mw?.value||"Horizontal") || "1";
        }
        return promptToRatio(pw?.value||"", mw?.value||"Horizontal") || "1";
    };
    addRatioCanvas(node, _getRatio, () => {
        return node.widgets?.find(w => w.name === "divide_mode")?.value || "Horizontal";
    });

    // _rpAutoUpdate 정의
    node._rpAutoUpdate = () => {
        const pw = node.widgets?.find(w => w.name === "prompt");
        const dw = node.widgets?.find(w => w.name === "divide_ratio");
        const mw = node.widgets?.find(w => w.name === "divide_mode");
        const aw = node.widgets?.find(w => w.name === "auto_div_calc");
        if (aw?.value !== "manual" && pw && dw) {
            const calc = promptToRatio(pw.value||"", mw?.value||"Horizontal");
            if (dw.value !== calc) dw.value = calc;
        }
        node.setDirtyCanvas(true);
    };

    // 콜백 연결 (중복 방지)
    const pw = node.widgets?.find(w => w.name === "prompt");
    if (pw && !pw._rpCallbackAttached) {
        let _t = null;
        const orig = pw.callback;
        pw.callback = v => { if (orig) orig.call(pw, v); clearTimeout(_t); _t = setTimeout(() => node._rpAutoUpdate(), 150); };
        pw._rpCallbackAttached = true;
    }
    const mw = node.widgets?.find(w => w.name === "divide_mode");
    if (mw && !mw._rpCallbackAttached) {
        const orig = mw.callback;
        mw.callback = v => { if (orig) orig.call(mw, v); node._rpAutoUpdate(); };
        mw._rpCallbackAttached = true;
    }
    const aw = node.widgets?.find(w => w.name === "auto_div_calc");
    if (aw && !aw._rpCallbackAttached) {
        const orig = aw.callback;
        aw.callback = v => {
            if (orig) orig.call(aw, v);
            if (v === "auto") {
                const pw2 = node.widgets?.find(w => w.name === "prompt");
                const dw2 = node.widgets?.find(w => w.name === "divide_ratio");
                const mw2 = node.widgets?.find(w => w.name === "divide_mode");
                if (pw2 && dw2) dw2.value = promptToRatio(pw2.value||"", mw2?.value||"Horizontal") || "";
            }
            node.setDirtyCanvas(true);
        };
        aw._rpCallbackAttached = true;
    }

    node.onRemoved = function() {
        for (const w of node.widgets||[]) if (w.canvas) w.canvas.remove?.();
    };

    // 크기 설정 및 초기 갱신
    requestAnimationFrame(() => {
        if (!node._rpInitSized) { node._rpInitSized = true; node.size[1] = 560; }
        node._rpAutoUpdate();
        node.setDirtyCanvas(true);
    });
}


app.registerExtension({
    name: "regional_prompter.nodes",

    async setup() {
        // ComfyUI 0.17: beforeRegisterNodeDef runs before nodes are registered,
        // but our extension loads AFTER nodes are already registered.
        // So we patch prototypes directly here in setup().
        const ppType = LiteGraph.registered_node_types["RPPromptParser"];
        if (ppType) {
            // Patch prototype for future nodes (new node creation)
            const _origCreated = ppType.prototype.onNodeCreated;
            ppType.prototype.onNodeCreated = function() {
                if (_origCreated) _origCreated.apply(this, arguments);
                _initPP(this);
            };
            const _origConfigure = ppType.prototype.onConfigure;
            ppType.prototype.onConfigure = function(info) {
                if (_origConfigure) _origConfigure.apply(this, arguments);
                _initPP(this);
            };
        }

        // Initialize existing nodes already in graph
        requestAnimationFrame(() => {
            (app.graph?._nodes || []).forEach(n => {
                if (n.type === "RPPromptParser") _initPP(n);
            });
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // ══════════════════════════════════════════
        // RPPromptParser
        // ══════════════════════════════════════════
        if (nodeData.name === "RPPromptParser") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onCreated) onCreated.apply(this, arguments);
                _initPP(this);
            };

            // onConfigure: called when workflow is loaded from file
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (origConfigure) origConfigure.call(this, info);
                _initPP(this);
            };
        }

        // ══════════════════════════════════════════
        // ══════════════════════════════════════════
        // RPRegionalDetailer — widgets_values 검증
        // scale_factor("x1"~"x4") → scale_to_pixel(INT 1024) 변경 마이그레이션
        // 실제 widgets_values (seed_control 포함):
        //   [16] scale_to_pixel(INT) ← 이전: scale_factor(COMBO "x2")
        // ══════════════════════════════════════════
        if (nodeData.name === "RPRegionalDetailer" ||
            nodeData.name === "RPRegionalDetailerZImage" ||
            nodeData.name === "RPRegionalDetailerQwen") {
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (origConfigure) origConfigure.call(this, info);
                const wv = info?.widgets_values;
                if (!Array.isArray(wv) || wv.length < 17) return;

                // wv[16] = scale_to_pixel(INT) 여야 함
                // string("x1"~"x4")이면 이전 scale_factor → 1024로 교체
                if (typeof wv[16] === "string" &&
                    ["x1","x2","x3","x4"].includes(wv[16])) {
                    wv[16] = 1024;
                    (this.widgets || []).forEach((w, i) => { if (i < wv.length) w.value = wv[i]; });
                }
            };
        }


        // RPKSampler — no wv migration hook needed
        // All widgets are in required block with fixed positions.

        // RPPromptParser — onConfigure: attach callbacks on existing node load
        if (nodeData.name === "RPPromptParser") {
            const origConfigure2 = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (origConfigure2) origConfigure2.call(this, info);
                // Attach callbacks after configure (safe: guarded by _rpCallbackAttached)
                const self = this;
                requestAnimationFrame(() => {
                    // prompt debounce
                    const pw = self.widgets?.find(w => w.name === "prompt");
                    if (pw && !pw._rpCallbackAttached) {
                        let _t = null;
                        const orig = pw.callback;
                        pw.callback = v => { if (orig) orig.call(pw, v); clearTimeout(_t); _t = setTimeout(() => self._rpAutoUpdate?.(), 150); };
                        pw._rpCallbackAttached = true;
                    }
                    // divide_mode immediate
                    const mw = self.widgets?.find(w => w.name === "divide_mode");
                    if (mw && !mw._rpCallbackAttached) {
                        const orig = mw.callback;
                        mw.callback = v => { if (orig) orig.call(mw, v); self._rpAutoUpdate?.(); };
                        mw._rpCallbackAttached = true;
                    }
                    // auto_div_calc
                    const aw = self.widgets?.find(w => w.name === "auto_div_calc");
                    if (aw && !aw._rpCallbackAttached) {
                        const orig = aw.callback;
                        aw.callback = v => {
                            if (orig) orig.call(aw, v);
                            if (v === "auto") {
                                const pw2 = self.widgets?.find(w => w.name === "prompt");
                                const dw2 = self.widgets?.find(w => w.name === "divide_ratio");
                                const mw2 = self.widgets?.find(w => w.name === "divide_mode");
                                if (pw2 && dw2) dw2.value = promptToRatio(pw2.value||"", mw2?.value||"Horizontal") || "";
                            }
                            self.setDirtyCanvas(true);
                        };
                        aw._rpCallbackAttached = true;
                    }
                    if (self._rpAutoUpdate) self._rpAutoUpdate();
                });
            };
        }


        // ══════════════════════════════════════════
        // RPKSampler — widgets_values 검증
        // 현재 위젯 순서:
        //   [0]seed  [1]steps  [2]cfg  [3]sampler_name  [4]scheduler
        //   [5]denoise  [6]use_base  [7]use_common  [8]base_ratio
        //   [9]lora_weight_adj  [10]debug
        // ══════════════════════════════════════════
        // ══════════════════════════════════════════
        // RPKSampler — widgets_values 검증
        // 정상 저장 구조 (seed_control 포함, 총 12개):
        //   [0]seed  [1]seed_control("randomize"/"fixed")
        //   [2]steps  [3]cfg  [4]sampler_name  [5]scheduler
        //   [6]denoise(FLOAT)  [7]use_base(BOOL)  [8]use_common(BOOL)
        //   [9]base_ratio(STRING)  [10]lora_weight_adj(INT)  [11]debug(BOOL)
        // ══════════════════════════════════════════
        if (nodeData.name === "RPKSampler") {
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (origConfigure) origConfigure.call(this, info);
                const wv = info?.widgets_values;
                if (!Array.isArray(wv) || wv.length < 6) return;

                // wv[6] = denoise(FLOAT) 여야 함
                // wv[6]이 boolean → denoise 누락된 구버전 → denoise=1.0 삽입
                if (typeof wv[6] === "boolean") {
                    wv.splice(6, 0, 1.0);
                }

                // wv[7] = use_base(BOOL) 여야 함
                // wv[7]이 string → use_base/use_common 없던 구버전 → 삽입
                if (wv.length >= 8 && typeof wv[7] !== "boolean") {
                    wv.splice(7, 0, false, true);
                }

                (this.widgets || []).forEach((w, i) => { if (i < wv.length) w.value = wv[i]; });
            };
        }

        // ══════════════════════════════════════════
        // RPKSamplerZImage — widgets_values 검증
        // 정상 순서: ..., shift(FLOAT>1), denoise(FLOAT 0~1), lora(INT), debug(BOOL)
        // ══════════════════════════════════════════
        if (nodeData.name === "RPKSamplerZImage") {
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (origConfigure) origConfigure.call(this, info);
                const wv = info?.widgets_values;
                if (!Array.isArray(wv) || wv.length < 4) return;

                // seed_control 위치 탐색
                let seedCtrlIdx = -1;
                for (let i = 0; i < Math.min(wv.length, 5); i++) {
                    if (wv[i] === "randomize" || wv[i] === "fixed") {
                        seedCtrlIdx = i; break;
                    }
                }
                if (seedCtrlIdx < 0) return;

                // steps,cfg,sampler,scheduler 다음이 shift 자리
                const shiftIdx = seedCtrlIdx + 1 + 4;
                if (shiftIdx >= wv.length) return;

                const shiftVal  = wv[shiftIdx];
                const denoiseVal = wv[shiftIdx + 1];

                // Case A: shift 자리에 0~1 float → shift 누락 → 삽입
                if (typeof shiftVal === "number" && shiftVal >= 0 && shiftVal <= 1.0) {
                    wv.splice(shiftIdx, 0, 3.0);
                }
                // Case B: shift 자리는 정상(>1)인데, denoise 자리가 INT(lora 값) → denoise 누락 → 1.0 삽입
                else if (typeof shiftVal === "number" && shiftVal > 1.0 &&
                         typeof denoiseVal === "number" && Number.isInteger(denoiseVal) && denoiseVal > 1) {
                    wv.splice(shiftIdx + 1, 0, 1.0);
                }

                (this.widgets || []).forEach((w, i) => { if (i < wv.length) w.value = wv[i]; });
            };
        }

        // ══════════════════════════════════════════
        // RPKSamplerQwen — widgets_values 검증 (ZImage와 동일 구조)
        // ══════════════════════════════════════════
        if (nodeData.name === "RPKSamplerQwen") {
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (origConfigure) origConfigure.call(this, info);
                const wv = info?.widgets_values;
                if (!Array.isArray(wv) || wv.length < 4) return;

                let seedCtrlIdx = -1;
                for (let i = 0; i < Math.min(wv.length, 5); i++) {
                    if (wv[i] === "randomize" || wv[i] === "fixed") {
                        seedCtrlIdx = i; break;
                    }
                }
                if (seedCtrlIdx < 0) return;

                const shiftIdx  = seedCtrlIdx + 1 + 4;
                if (shiftIdx >= wv.length) return;

                const shiftVal   = wv[shiftIdx];
                const denoiseVal = wv[shiftIdx + 1];

                if (typeof shiftVal === "number" && shiftVal >= 0 && shiftVal <= 1.0) {
                    wv.splice(shiftIdx, 0, 3.0);
                } else if (typeof shiftVal === "number" && shiftVal > 1.0 &&
                           typeof denoiseVal === "number" && Number.isInteger(denoiseVal) && denoiseVal > 1) {
                    wv.splice(shiftIdx + 1, 0, 1.0);
                }

                (this.widgets || []).forEach((w, i) => { if (i < wv.length) w.value = wv[i]; });
            };
        }

        // RPRatioParser — handled via patchRPRatioParser() below (same pattern as RPPromptParser)
    },

    // 그래프 실행 완료 후 모든 RP 노드 갱신
    async afterRegisterNodeDef() {},

    settings: [{
        id: "regional_prompter.show_preview",
        name: "비율 미리보기 표시",
        type: "boolean",
        defaultValue: true,
    }],

    aboutPageBadges: [{
        label: "Regional Prompter (원본 SD-WebUI)",
        url: "https://github.com/hako-mikan/sd-webui-regional-prompter",
        icon: "pi pi-external-link",
    }],
});

// 실행 완료 시 RPRatioParser 시각화 강제 갱신
app.api?.addEventListener("executed", () => {
    const nodes = app.graph?._nodes || [];
    nodes.forEach(n => {
        if (n.type === "RPRatioParser") {
            n.setDirtyCanvas(true);
        }
    });
});

// ── RPRatioParser: prototype 패치 + 기존 노드 초기화 ─────────────────────────
// Same pattern as RPPromptParser — ComfyUI 0.17 does not call setup() reliably.
function _initRP(node) {
    // Guard: skip if canvas widget already exists
    if (node.widgets?.some(w => w.name === "rp-ratio-canvas")) return;

    const canvasWidget = addRatioCanvas(
        node,
        () => {
            const linked = getLinkedValue(node, "divide_ratio");
            if (linked && linked !== "auto") return linked;
            const w = node.widgets?.find(w => w.name === "divide_ratio" || w.name === "aratios");
            return String(w?.value || "1,1");
        },
        () => {
            const linked = getLinkedValue(node, "divide_mode");
            if (linked) return linked;
            const w = node.widgets?.find(w => w.name === "divide_mode" || w.name === "mode");
            return w?.value || "Horizontal";
        }
    );

    if (canvasWidget) {
        // Show canvas only when divide_ratio input is connected
        canvasWidget.computeSize = (width) => {
            const inp = node.inputs?.find(i => i.name === "divide_ratio");
            return (inp?.link != null) ? [width || 0, CANVAS_H] : [width || 0, 0];
        };
    }

    // onExecuted: refresh visualization after graph execution
    node.onExecuted = function(output) {
        this.setDirtyCanvas(true);
    };

    // onConnectionsChange: show/hide canvas when link connects/disconnects
    node.onConnectionsChange = function() {
        this.setDirtyCanvas(true);
    };

    node.onRemoved = function() {
        for (const w of node.widgets || []) if (w.canvas) w.canvas.remove?.();
    };
}

(function patchRPRatioParser() {
    const rpType = LiteGraph.registered_node_types?.["RPRatioParser"];
    if (rpType) {
        if (!rpType.prototype._rpRatioPatched) {
            rpType.prototype._rpRatioPatched = true;
            const _origCreated = rpType.prototype.onNodeCreated;
            rpType.prototype.onNodeCreated = function() {
                if (_origCreated) _origCreated.apply(this, arguments);
                _initRP(this);
            };
            const _origConfigure = rpType.prototype.onConfigure;
            rpType.prototype.onConfigure = function(info) {
                if (_origConfigure) _origConfigure.apply(this, arguments);
                _initRP(this);
            };
        }
        // Initialize existing nodes already in graph
        requestAnimationFrame(() => {
            (app.graph?._nodes || []).forEach(n => {
                if (n.type === "RPRatioParser") _initRP(n);
            });
        });
    } else {
        let tries = 0;
        const timer = setInterval(() => {
            const t = LiteGraph.registered_node_types?.["RPRatioParser"];
            if (t) { clearInterval(timer); patchRPRatioParser(); }
            else if (++tries > 30) clearInterval(timer);
        }, 100);
    }
})();

// ComfyUI 0.17에서 setup()이 호출되지 않으므로 여기서 직접 실행
(function patchRPPromptParser() {
    const ppType = LiteGraph.registered_node_types?.["RPPromptParser"];
    if (ppType) {
        // prototype 패치 (신규 노드 생성 시 적용)
        const _origCreated = ppType.prototype.onNodeCreated;
        if (!ppType.prototype._rpPatched) {
            ppType.prototype._rpPatched = true;
            ppType.prototype.onNodeCreated = function() {
                if (_origCreated) _origCreated.apply(this, arguments);
                _initPP(this);
            };
            const _origConfigure = ppType.prototype.onConfigure;
            ppType.prototype.onConfigure = function(info) {
                if (_origConfigure) _origConfigure.apply(this, arguments);
                _initPP(this);
            };
        }
        // 이미 그래프에 있는 노드 즉시 초기화
        requestAnimationFrame(() => {
            (app.graph?._nodes || []).forEach(n => {
                if (n.type === "RPPromptParser") _initPP(n);
            });
        });
    } else {
        // LiteGraph에 아직 없으면 재시도 (최대 3초)
        let tries = 0;
        const timer = setInterval(() => {
            const t = LiteGraph.registered_node_types?.["RPPromptParser"];
            if (t) { clearInterval(timer); patchRPPromptParser(); }
            else if (++tries > 30) clearInterval(timer);
        }, 100);
    }
})();

} // end _registerRPCast
