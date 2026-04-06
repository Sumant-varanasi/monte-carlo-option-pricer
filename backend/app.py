# app.py — flask server, thin wrapper around engine.py
# serves the API endpoints and the frontend HTML in one go

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import os

from engine import (
    black_scholes,
    simulate_gbm_paths,
    mc_standard,
    mc_antithetic,
    mc_control_variate,
    sensitivity_spot,
    sensitivity_vol,
    sensitivity_time,
    build_histogram,
)

app = Flask(__name__)
CORS(app)

FRONTEND_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Monte Carlo Option Pricer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#060a0e;color:#c8e6d8;font-family:'IBM Plex Mono',monospace;min-height:100vh}
.w{max-width:1100px;margin:0 auto;padding:24px 20px}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0a0e12}::-webkit-scrollbar-thumb{background:#1a3a28;border-radius:3px}
.sub{font-size:9px;letter-spacing:.5em;color:#00ff88;text-transform:uppercase;margin-bottom:6px;opacity:.7}
h1{font-size:30px;font-weight:300;color:#00ff88;text-shadow:0 0 40px rgba(0,255,136,.15);margin:0}
.tag{font-size:12px;color:#3a6a50;margin-top:5px;font-style:italic}
.el{font-size:10px;color:#2a5a40;margin-top:6px}
.glow{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:280px;height:280px;background:radial-gradient(circle,rgba(0,255,136,.03) 0%,transparent 70%);border-radius:50%;pointer-events:none}
.err{background:rgba(255,60,60,.08);border:1px solid rgba(255,60,60,.3);border-radius:6px;padding:12px;margin-bottom:16px;text-align:center;font-size:12px;color:#ff6666;display:none}
.pm{background:rgba(0,255,136,.02);border:1px solid rgba(0,255,136,.08);border-radius:8px;padding:14px;margin-bottom:16px}
.pg{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px}
.lb{font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.12em;color:#4a7a60;margin-bottom:2px;display:block}
.ip{background:rgba(0,255,136,.04);border:1px solid rgba(0,255,136,.15);border-radius:3px;padding:6px 8px;color:#c8e6d8;font-family:inherit;font-size:12px;width:100%;outline:none}
.ip:focus{border-color:rgba(0,255,136,.4)}
.bt{background:rgba(0,255,136,.12);color:#00ff88;border:1px solid rgba(0,255,136,.3);border-radius:4px;padding:9px 36px;font-size:11px;font-weight:500;letter-spacing:.12em;text-transform:uppercase;cursor:pointer;font-family:inherit}
.bt:hover{background:rgba(0,255,136,.18)}.bt:disabled{background:rgba(0,255,136,.06);color:#3a6a50;cursor:wait}
.cds{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:16px}
.cd{background:rgba(0,0,0,.3);border-radius:6px;padding:12px 14px;position:relative;overflow:hidden}
.cl{position:absolute;top:0;left:0;right:0;height:1px}
.cv{font-size:22px;font-weight:300}.ce{font-size:10px;color:#3a6a50;margin-top:2px}
.vb{background:rgba(0,0,0,.2);border:1px solid rgba(0,255,136,.06);border-radius:6px;padding:12px;margin-bottom:16px}
.vg{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}
.vr{display:flex;justify-content:space-between;font-size:10px;color:#4a7a60;margin-bottom:2px}
.vbb{height:4px;background:rgba(255,255,255,.03);border-radius:2px;overflow:hidden}
.vbi{height:100%;border-radius:2px}
.ts{display:flex;gap:1px;flex-wrap:wrap}.tb{border-radius:6px 6px 0 0;padding:7px 16px;font-size:10px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;cursor:pointer;border:1px solid rgba(255,255,255,.03);background:rgba(0,0,0,.2);color:#3a6a50;font-family:inherit}
.tb:hover{color:#00cc66}.tb.a{background:rgba(0,255,136,.06);border-color:rgba(0,255,136,.2);border-bottom-color:transparent;color:#00ff88}
.ca{background:rgba(0,0,0,.2);border:1px solid rgba(0,255,136,.06);border-radius:0 6px 6px 6px;padding:14px;min-height:400px}
.cd2{font-size:11px;color:#3a6a50;margin-bottom:8px;font-style:italic}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:6px;margin-top:8px;padding:8px 12px;background:rgba(0,0,0,.2);border-radius:4px}
.ft{margin-top:16px;padding:12px 14px;background:rgba(0,0,0,.15);border:1px solid rgba(0,255,136,.05);border-radius:6px;font-size:10px;color:#3a6a50;line-height:1.5}
canvas{max-height:380px}
.sp{display:inline-block;width:12px;height:12px;border:2px solid #00ff8833;border-top-color:#00ff88;border-radius:50%;animation:sp .8s linear infinite;margin-right:6px;vertical-align:middle}
@keyframes sp{to{transform:rotate(360deg)}}
.h{display:none!important}
.sg2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
</style>
</head>
<body>
<div class="w">
<div style="text-align:center;margin-bottom:24px;position:relative"><div class="glow"></div>
<div class="sub">Python · NumPy · Flask · Chart.js</div>
<h1>Monte Carlo Option Pricer</h1>
<div class="tag">GBM Simulation · Black-Scholes Validation · Variance Reduction</div>
<div class="el" id="el"></div></div>
<div class="err" id="er"></div>
<div class="pm"><div class="pg">
<div><label class="lb">Spot (S0)</label><input class="ip" type="number" id="S0" value="100" step="1"></div>
<div><label class="lb">Strike (K)</label><input class="ip" type="number" id="K" value="105" step="1"></div>
<div><label class="lb">Maturity (T)</label><input class="ip" type="number" id="T" value="1" step="0.1"></div>
<div><label class="lb">Rate (r)</label><input class="ip" type="number" id="r" value="0.05" step="0.005"></div>
<div><label class="lb">Vol</label><input class="ip" type="number" id="sigma" value="0.2" step="0.01"></div>
<div><label class="lb">Simulations</label><input class="ip" type="number" id="n_sim" value="50000" step="10000"></div>
<div><label class="lb">Paths</label><input class="ip" type="number" id="n_paths" value="30" step="5" min="5" max="100"></div>
<div><label class="lb">Type</label><select class="ip" id="option_type"><option value="call">Call</option><option value="put">Put</option></select></div>
</div><div style="text-align:center;margin-top:12px"><button class="bt" id="rb" onclick="go()">&#9654; Run Simulation</button></div></div>
<div id="ra" class="h">
<div class="cds" id="cds"></div>
<div class="vb" id="vb"></div>
<div class="ts" id="ts"></div>
<div class="ca">
<div id="pp"><div class="cd2" id="pd"></div><canvas id="c1"></canvas><div style="text-align:center;margin-top:4px;font-size:9px;color:#3a6a50">White dashed = Mean · Green dashed = Strike</div></div>
<div id="pf" class="h"><div class="cd2">Discounted payoff distribution</div><canvas id="c2"></canvas><div class="sg" id="st"></div></div>
<div id="pc" class="h"><div class="cd2">MC methods converging toward Black-Scholes</div><canvas id="c3"></canvas><div id="bl" style="text-align:center;font-size:10px;color:#00ff88;margin-top:4px"></div></div>
<div id="ps" class="h"><div class="cd2" id="sd"></div><canvas id="c4"></canvas><div class="sg2"><canvas id="c5"></canvas><canvas id="c6"></canvas></div></div>
</div>
<div class="ft">
<div style="color:#00ff88;font-size:9px;letter-spacing:.15em;text-transform:uppercase;margin-bottom:4px">Architecture</div>
<p style="margin:0 0 4px"><span style="color:#00d4ff">Backend:</span> Python + Flask + NumPy + SciPy</p>
<p style="margin:0 0 4px"><span style="color:#ffaa00">API:</span> /api/simulate · /api/sensitivity</p>
<p style="margin:0"><span style="color:#ff6688">Frontend:</span> Chart.js · vanilla JS</p>
</div></div></div>
<script>
var API=location.origin,C={},D=Chart.defaults;D.color="#4a7a60";D.borderColor="rgba(0,255,136,.06)";D.font.family="'IBM Plex Mono',monospace";D.font.size=10;
function gv(i){return document.getElementById(i).value}
function gp(){return{S0:+gv("S0"),K:+gv("K"),T:+gv("T"),r:+gv("r"),sigma:+gv("sigma"),n_sim:+gv("n_sim"),n_paths:+gv("n_paths"),option_type:gv("option_type")}}
async function go(){var b=document.getElementById("rb");b.disabled=true;b.innerHTML='<span class="sp"></span>Computing...';document.getElementById("er").style.display="none";var p=gp();
try{var[s,e]=await Promise.all([fetch(API+"/api/simulate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)}),fetch(API+"/api/sensitivity",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)})]);
if(!s.ok)throw new Error(s.status);if(!e.ok)throw new Error(e.status);rn(await s.json(),await e.json(),p)}catch(x){var er=document.getElementById("er");er.textContent=x.message+" - Is Flask running?";er.style.display="block"}finally{b.disabled=false;b.textContent="\u25b6 Run Simulation"}}
function rn(R,S,P){document.getElementById("ra").classList.remove("h");document.getElementById("el").textContent="Computed in "+R.elapsed_ms+"ms";
var c=[{l:"Black-Scholes",v:R.bs_price,c:"#00ff88",e:null},{l:"MC Standard",v:R.mc_standard.price,c:"#00d4ff",e:R.mc_standard.stderr},{l:"MC Antithetic",v:R.mc_antithetic.price,c:"#ffaa00",e:R.mc_antithetic.stderr},{l:"MC Control Var.",v:R.mc_control.price,c:"#ff6688",e:R.mc_control.stderr}];
document.getElementById("cds").innerHTML=c.map(function(x){var p=x.e!==null?((x.v-R.bs_price)/R.bs_price*100).toFixed(3):"";return'<div class="cd" style="border:1px solid '+x.c+'18"><div class="cl" style="background:linear-gradient(90deg,transparent,'+x.c+'44,transparent)"></div><div class="lb" style="color:'+x.c+'88">'+x.l+'</div><div class="cv" style="color:'+x.c+'">'+x.v.toFixed(4)+'</div>'+(x.e!==null?'<div class="ce">\xb1'+x.e.toFixed(4)+' <span style="color:'+x.c+'66">('+p+'%)</span></div>':'')+'</div>'}).join("");
var vr=[{l:"Standard MC",s:R.mc_standard.stderr,c:"#00d4ff"},{l:"Antithetic",s:R.mc_antithetic.stderr,c:"#ffaa00"},{l:"Control Var.",s:R.mc_control.stderr,c:"#ff6688"}];
document.getElementById("vb").innerHTML='<div class="lb" style="color:#00ff88;margin-bottom:8px">Variance Reduction</div><div class="vg">'+vr.map(function(v){var r=(1-v.s/R.mc_standard.stderr)*100,w=v.s/R.mc_standard.stderr*100;return'<div><div class="vr"><span>'+v.l+'</span><span style="color:'+v.c+'">'+(r>0?"\u2212"+r.toFixed(1)+"%":"baseline")+'</span></div><div class="vbb"><div class="vbi" style="width:'+Math.min(w,100)+'%;background:linear-gradient(90deg,'+v.c+','+v.c+'66)"></div></div></div>'}).join("")+'</div>';
document.getElementById("ts").innerHTML=["paths","payoff","convergence","sensitivity"].map(function(t,i){var n={paths:"GBM Paths",payoff:"Payoff",convergence:"Convergence",sensitivity:"Sensitivity"};return'<button class="tb'+(i===0?" a":"")+'" data-t="'+t+'" onclick="sw(\''+t+'\')">'+n[t]+'</button>'}).join("");
dp(R,P);df(R);dc(R);ds(S,P);sw("paths")}
function sw(id){["pp","pf","pc","ps"].forEach(function(x,i){document.getElementById(x).classList.toggle("h",["paths","payoff","convergence","sensitivity"][i]!==id)});document.querySelectorAll(".tb").forEach(function(b){b.classList.toggle("a",b.dataset.t===id)})}
function dx(i){if(C[i]){C[i].destroy();delete C[i]}}
function dp(R,P){dx("p");var g=R.gbm_paths,l=g.time_grid.filter(function(_,i){return i%10===0||i===g.time_grid.length-1}).map(function(t){return t.toFixed(2)});
var pc=["#00ff88","#00e67a","#00cc6c","#00b35e","#009950","#00804a","#33ff99","#1aff8e","#00d4ff","#00bbdd","#00a3bb","#33ddff","#00ffcc","#00e6b8","#00cca3","#33ffcc","#00c8ee","#4de5ff","#007177","#008a99","#00b38f","#00997a","#00663c","#66ffd9","#4dffcc","#00f0bf","#66eaff","#1ad5ff","#4dffa6","#1affcc"];
var ds=g.paths.map(function(p,i){var d=p.filter(function(_,j){return j%10===0||j===p.length-1});return{data:d,borderColor:pc[i%pc.length],borderWidth:.7,pointRadius:0,tension:.1,fill:false,label:""}});
var mp=g.mean_path.filter(function(_,i){return i%10===0||i===g.mean_path.length-1});
ds.push({data:mp,borderColor:"#fff",borderWidth:2,pointRadius:0,borderDash:[6,3],fill:false,label:"Mean"});
document.getElementById("pd").textContent=P.n_paths+" trajectories \u2014 risk-neutral GBM via NumPy";
C.p=new Chart(document.getElementById("c1"),{type:"line",data:{labels:l,datasets:ds},options:{responsive:true,animation:false,plugins:{legend:{display:false},tooltip:{enabled:false}},scales:{x:{title:{display:true,text:"Time (years)",color:"#3a6a50"},ticks:{maxTicksLimit:10}},y:{title:{display:true,text:"Price",color:"#3a6a50"}}}}})}
function df(R){dx("f");var h=R.histogram,l=h.map(function(b){return b.range}),d=h.map(function(b){return b.count});
C.f=new Chart(document.getElementById("c2"),{type:"bar",data:{labels:l,datasets:[{data:d,backgroundColor:h.map(function(b){return b.midpoint===0?"rgba(212,175,55,.5)":"rgba(0,212,255,.3)"}),borderColor:h.map(function(b){return b.midpoint===0?"#d4af37":"#00d4ff"}),borderWidth:.5}]},options:{responsive:true,animation:false,plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:"Discounted Payoff",color:"#3a6a50"},ticks:{maxTicksLimit:10}},y:{title:{display:true,text:"Frequency",color:"#3a6a50"}}}}});
document.getElementById("st").innerHTML=[{l:"Mean",v:R.mc_standard.price.toFixed(4)},{l:"ITM Prob",v:R.stats.itm_prob.toFixed(1)+"%"},{l:"Max",v:R.stats.max_payoff.toFixed(2)},{l:"Std Dev",v:R.stats.std_dev.toFixed(4)}].map(function(s){return'<div><div style="font-size:8px;color:#3a6a50;text-transform:uppercase;letter-spacing:.08em">'+s.l+'</div><div style="font-size:13px;color:#c8e6d8;margin-top:1px">'+s.v+'</div></div>'}).join("")}
function dc(R){dx("c");var d=R.convergence,l=d.map(function(x){return x.n>=1e3?(x.n/1e3|0)+"k":x.n});
C.c=new Chart(document.getElementById("c3"),{type:"line",data:{labels:l,datasets:[{data:d.map(function(x){return x.standard}),borderColor:"#00d4ff",label:"Standard",borderWidth:1.5,pointRadius:0,tension:.2,fill:false},{data:d.map(function(x){return x.antithetic}),borderColor:"#ffaa00",label:"Antithetic",borderWidth:1.5,pointRadius:0,tension:.2,fill:false},{data:d.map(function(x){return x.control}),borderColor:"#ff6688",label:"Control Var.",borderWidth:1.5,pointRadius:0,tension:.2,fill:false}]},options:{responsive:true,animation:false,plugins:{legend:{labels:{color:"#4a7a60",usePointStyle:true,pointStyle:"line"}}},scales:{x:{title:{display:true,text:"Simulations",color:"#3a6a50"},ticks:{maxTicksLimit:10}},y:{title:{display:true,text:"Option Price",color:"#3a6a50"}}}}});
document.getElementById("bl").textContent="-- Black-Scholes = "+R.bs_price.toFixed(4)}
function ds(S,P){dx("s");dx("v");dx("t");document.getElementById("sd").textContent="BS "+P.option_type+" sensitivity \u2014 Python backend";
C.s=new Chart(document.getElementById("c4"),{type:"line",data:{labels:S.spot.spots.map(function(s){return s.toFixed(0)}),datasets:[{data:S.spot.intrinsic,borderColor:"#00ff8833",borderDash:[4,4],borderWidth:1,pointRadius:0,fill:false,label:"Intrinsic",tension:.1},{data:S.spot.prices,borderColor:"#00d4ff",borderWidth:2,pointRadius:0,fill:false,label:"BS Price",tension:.2}]},options:{responsive:true,animation:false,plugins:{legend:{labels:{color:"#4a7a60",usePointStyle:true,pointStyle:"line"}}},scales:{x:{title:{display:true,text:"Spot",color:"#3a6a50"},ticks:{maxTicksLimit:8}},y:{title:{display:true,text:"Price",color:"#3a6a50"}}}}});
C.v=new Chart(document.getElementById("c5"),{type:"line",data:{labels:S.vol.vols.map(function(v){return v.toFixed(0)}),datasets:[{data:S.vol.prices,borderColor:"#ffaa00",borderWidth:2,pointRadius:0,tension:.2,fill:false}]},options:{responsive:true,animation:false,plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:"Vol %",color:"#3a6a50"},ticks:{maxTicksLimit:6}},y:{title:{display:true,text:"Price",color:"#3a6a50"}}}}});
C.t=new Chart(document.getElementById("c6"),{type:"line",data:{labels:S.time.times.map(function(t){return t.toFixed(2)}),datasets:[{data:S.time.prices,borderColor:"#ff6688",borderWidth:2,pointRadius:0,tension:.2,fill:false}]},options:{responsive:true,animation:false,plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:"Time (yrs)",color:"#3a6a50"},ticks:{maxTicksLimit:6}},y:{title:{display:true,text:"Price",color:"#3a6a50"}}}}})}
window.addEventListener("load",function(){go()});
</script>
</body></html>'''

@app.route("/")
def serve_frontend():
    return FRONTEND_HTML


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "monte-carlo-option-pricer"})


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Main endpoint — runs all three MC methods + BS and returns everything."""
    data = request.get_json()

    S0 = float(data.get("S0", 100))
    K = float(data.get("K", 105))
    T = float(data.get("T", 1.0))
    r = float(data.get("r", 0.05))
    sigma = float(data.get("sigma", 0.2))
    n_sim = int(data.get("n_sim", 50000))
    n_paths = int(data.get("n_paths", 30))
    option_type = data.get("option_type", "call")
    seed = data.get("seed", None)

    # don't let anyone ddos us with 10M sims
    n_sim = max(100, min(n_sim, 500000))
    n_paths = max(5, min(n_paths, 100))
    T = max(0.01, T)
    sigma = max(0.01, sigma)

    t0 = time.perf_counter()

    bs_price = black_scholes(S0, K, T, r, sigma, option_type)
    gbm = simulate_gbm_paths(S0, r, sigma, T, n_steps=252, n_paths=n_paths, seed=seed)

    # run all three MC flavors
    res_std = mc_standard(S0, K, T, r, sigma, n_sim, option_type, seed=seed)
    res_anti = mc_antithetic(S0, K, T, r, sigma, n_sim, option_type, seed=seed)
    res_ctrl = mc_control_variate(S0, K, T, r, sigma, n_sim, option_type, seed=seed)

    histogram = build_histogram(res_std.payoffs, bins=60)

    # quick stats for the payoff panel
    import numpy as np
    payoffs_arr = np.array(res_std.payoffs)
    itm_prob = float(np.mean(payoffs_arr > 0) * 100)
    max_payoff = float(payoffs_arr.max())
    std_dev = float(payoffs_arr.std())

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # merge convergence series so the frontend can overlay them on one chart
    conv_map = {}
    for c in res_std.convergence:
        conv_map[c["n"]] = {
            "n": c["n"],
            "standard": c["price"],
            "stdErr": c["stderr"],
            "bsPrice": bs_price,
        }
    for c in res_anti.convergence:
        entry = conv_map.get(c["n"], {"n": c["n"], "bsPrice": bs_price})
        entry["antithetic"] = c["price"]
        entry["antiErr"] = c["stderr"]
        conv_map[c["n"]] = entry
    for c in res_ctrl.convergence:
        entry = conv_map.get(c["n"], {"n": c["n"], "bsPrice": bs_price})
        entry["control"] = c["price"]
        entry["ctrlErr"] = c["stderr"]
        conv_map[c["n"]] = entry

    convergence = sorted(conv_map.values(), key=lambda x: x["n"])

    return jsonify({
        "bs_price": bs_price,
        "mc_standard": {
            "price": res_std.price,
            "stderr": res_std.stderr,
        },
        "mc_antithetic": {
            "price": res_anti.price,
            "stderr": res_anti.stderr,
        },
        "mc_control": {
            "price": res_ctrl.price,
            "stderr": res_ctrl.stderr,
        },
        "gbm_paths": gbm,
        "histogram": histogram,
        "convergence": convergence,
        "stats": {
            "itm_prob": itm_prob,
            "max_payoff": max_payoff,
            "std_dev": std_dev,
        },
        "elapsed_ms": round(elapsed_ms, 1),
    })


@app.route("/api/sensitivity", methods=["POST"])
def sensitivity():
    """BS price sweeps across spot, vol, and time — for the sensitivity charts."""
    data = request.get_json()

    S0 = float(data.get("S0", 100))
    K = float(data.get("K", 105))
    T = float(data.get("T", 1.0))
    r = float(data.get("r", 0.05))
    sigma = float(data.get("sigma", 0.2))
    option_type = data.get("option_type", "call")

    return jsonify({
        "spot": sensitivity_spot(S0, K, T, r, sigma, option_type),
        "vol": sensitivity_vol(S0, K, T, r, sigma, option_type),
        "time": sensitivity_time(S0, K, T, r, sigma, option_type),
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  Monte Carlo Option Pricing — Backend Server")
    print("  Endpoints:")
    print("    POST /api/simulate")
    print("    POST /api/sensitivity")
    print("    GET  /api/health")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)

