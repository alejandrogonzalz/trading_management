import { useState, useEffect, useMemo } from 'react';
import { ScanSearch, TrendingUp, TrendingDown, Scale, Zap, Activity, Layers, Target, Info, ArrowUp, ArrowDown, Filter, X, ShieldCheck, Target as TargetIcon, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const AnalysisModal = ({ data, onClose, onQuickTrade }) => {
    if (!data) return null;

    const getBiasColor = (bias) => {
        const b = bias?.toLowerCase() || 'neutral';
        if (b === 'bullish') return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20';
        if (b === 'bearish') return 'text-rose-400 bg-rose-400/10 border-rose-400/20';
        return 'text-slate-400 bg-slate-400/10 border-slate-400/20';
    };

    return (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4 bg-slate-950/95 backdrop-blur-lg">
            <div className="bg-slate-900 w-full max-w-lg rounded-3xl border border-slate-800 shadow-[0_0_80px_rgba(0,0,0,0.8)] overflow-hidden">
                <div className="flex justify-between items-center p-6 border-b border-slate-800 bg-slate-900/50">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-500/10 rounded-xl">
                            <Activity size={20} className="text-blue-400" />
                        </div>
                        <div>
                            <h2 className="text-xl font-black text-white tracking-tighter uppercase">{data.pair || 'Analysis'}</h2>
                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">AI Expert Deep Dive</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-full transition-colors text-slate-500 hover:text-white"><X size={20} /></button>
                </div>

                <div className="p-6 space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                        <div className={`p-4 rounded-2xl border ${getBiasColor(data.bias)} flex flex-col items-center gap-1`}>
                            <span className="text-[10px] font-black uppercase tracking-widest opacity-60">AI Bias</span>
                            <span className="text-lg font-black uppercase">{data.bias || 'Neutral'}</span>
                        </div>
                        <div className="p-4 rounded-2xl border border-slate-800 bg-slate-950/50 flex flex-col items-center gap-1">
                            <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">Confidence</span>
                            <span className="text-lg font-black text-white">{data.confidence || '5'}/10</span>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <div className="flex items-center gap-2 text-blue-400">
                            <TargetIcon size={16} />
                            <span className="text-xs font-black uppercase tracking-widest">Trade Setup</span>
                        </div>
                        <div className="bg-slate-950/50 rounded-2xl border border-slate-800 p-4">
                            <p className="text-sm font-bold text-white mb-2">{data.trade_setup || 'No setup provided'}</p>
                            <div className="grid grid-cols-3 gap-2">
                                <div className="space-y-1">
                                    <span className="text-[9px] font-black text-slate-500 uppercase block">Entry</span>
                                    <span className="text-xs font-mono font-bold text-slate-300">${data.entry || '--'}</span>
                                </div>
                                <div className="space-y-1">
                                    <span className="text-[9px] font-black text-emerald-500/70 uppercase block">Take Profit</span>
                                    <span className="text-xs font-mono font-bold text-emerald-400">${data.tp || '--'}</span>
                                </div>
                                <div className="space-y-1">
                                    <span className="text-[9px] font-black text-rose-500/70 uppercase block">Stop Loss</span>
                                    <span className="text-xs font-mono font-bold text-rose-400">${data.sl || '--'}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <div className="flex items-center gap-2 text-purple-400">
                            <ShieldCheck size={16} />
                            <span className="text-xs font-black uppercase tracking-widest">Reasoning</span>
                        </div>
                        <p className="text-xs text-slate-400 leading-relaxed font-medium bg-slate-950/30 p-4 rounded-2xl italic border border-slate-800/50">"{data.reasoning || 'No analysis available'}"</p>
                    </div>
                </div>

                <div className="p-6 bg-slate-950/50 border-t border-slate-800 flex justify-between items-center">
                    <div className="flex items-center gap-2 text-slate-600">
                        <AlertCircle size={14} />
                        <span className="text-[9px] font-black uppercase tracking-widest">Risk Reward {data.risk_reward || 'N/A'}</span>
                    </div>
                    <button 
                        onClick={() => onQuickTrade(data)}
                        className="bg-blue-600 hover:bg-blue-500 text-white text-[10px] font-black px-6 py-2.5 rounded-xl transition-all shadow-lg shadow-blue-900/20 uppercase tracking-widest"
                    >
                        Auto-Trade Setup
                    </button>
                </div>
            </div>
        </div>
    );
};

const HeatmapCell = ({ heatmap }) => {
    if (!heatmap) return null;
    const getHeatmapIcon = (status) => {
        if (status === 'STRONG_BULLISH') return <TrendingUp size={10} className="text-emerald-400 stroke-[3]" />;
        if (status === 'BULLISH') return <TrendingUp size={10} className="text-emerald-500/70" />;
        if (status === 'STRONG_BEARISH') return <TrendingDown size={10} className="text-rose-400 stroke-[3]" />;
        if (status === 'BEARISH') return <TrendingDown size={10} className="text-rose-500/70" />;
        return <Scale size={10} className="text-slate-500" />;
    };
    const timeframeOrder = ['5m', '15m', '1h', '4h', '1d', '1w', '1M'];
    return (
        <div className="flex gap-1">
            {timeframeOrder.map(tf => {
                const status = heatmap[tf];
                if (!status) return null;
                return (<div key={tf} title={`${tf}: ${status}`} className="flex items-center justify-center w-5 h-5 bg-slate-800 rounded border border-slate-700">{getHeatmapIcon(status)}</div>);
            })}
        </div>
    );
};

const ColumnHeader = ({ label, tooltip, sortKey, currentSort, onSort }) => {
    const isSorted = currentSort.key === sortKey;
    return (
        <th className="py-5 px-4 group/head relative">
            <div className="flex items-center gap-1 cursor-pointer hover:text-blue-400 transition-colors" onClick={() => onSort(sortKey)}>
                <span className="flex items-center gap-1.5 whitespace-nowrap text-[10px] font-black uppercase tracking-[0.1em]">
                    {label}
                    {isSorted && (currentSort.direction === 'desc' ? <ArrowDown size={10} /> : <ArrowUp size={10} />)}
                </span>
                <Info size={10} className="text-slate-700 group-hover/head:text-blue-500 transition-colors" />
            </div>
            {/* Improved Tooltip positioning */}
            <div className="fixed mb-2 w-64 bg-slate-800 p-3 rounded-xl border border-slate-700 shadow-2xl z-[1000] invisible group-hover/head:visible pointer-events-none transition-all duration-200 opacity-0 group-hover/head:opacity-100 -translate-y-full ml-[-10px]">
                <p className="text-[10px] text-slate-200 font-black uppercase mb-1 tracking-wider border-b border-slate-700 pb-1">{label}</p>
                <p className="text-[10px] text-slate-400 font-bold normal-case leading-relaxed">{tooltip}</p>
            </div>
        </th>
    );
};

const ScannerView = ({ symbols, onAutoTrade }) => {
    const [scannerResults, setScannerResults] = useState([]);
    const [scanning, setScanning] = useState(false);
    const [rankingLLM, setRankingLLM] = useState(false);
    const [analyzingLLM, setAnalyzingLLM] = useState<string | null>(null);
    const [analysisResult, setAnalysisResult] = useState<any>(null);
    const [selectedTF, setSelectedTF] = useState('1h');
    const [lastScanTF, setLastScanTF] = useState('1h');
    const [error, setError] = useState(null);
    const [sortConfig, setSortConfig] = useState({ key: 'score', direction: 'desc' });
    const [filterText, setFilterText] = useState('');
    const navigate = useNavigate();

    const timeframes = ['5m', '15m', '1h', '4h', '1d', '1w', '1M'];

    const handleSort = (key) => {
        setSortConfig(prev => ({
            key,
            direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
        }));
    };

    const processedResults = useMemo(() => {
        if (!scannerResults) return [];
        let items = [...scannerResults];
        if (filterText) { items = items.filter(i => i.pair?.toLowerCase().includes(filterText.toLowerCase())); }
        items.sort((a, b) => {
            let valA = a[sortConfig.key];
            let valB = b[sortConfig.key];
            if (sortConfig.key === 'ai_rank') {
                valA = valA === 'N/A' || !valA ? 999 : parseInt(valA);
                valB = valB === 'N/A' || !valB ? 999 : parseInt(valB);
            }
            if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
            if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
        return items;
    }, [scannerResults, sortConfig, filterText]);

    const handleRunScan = async () => {
        setScanning(true);
        setError(null);
        try {
            const res = await fetch(`${API_BASE}/scanner/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pairs: [], timeframe: selectedTF })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Failed to run scanner");
            setScannerResults(data);
            setLastScanTF(selectedTF);
        } catch (err) { setError(err.message); } finally { setScanning(false); }
    };

    const handleRankLLM = async () => {
        setRankingLLM(true);
        setError(null);
        try {
            const res = await fetch(`${API_BASE}/llm/rank`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Failed to rank with LLM");
            setScannerResults(prevResults => {
                const newResults = prevResults.map(row => {
                    const normalizedPair = row.pair.trim().toUpperCase();
                    const aiMatchKey = Object.keys(data).find(k => k.trim().toUpperCase() === normalizedPair);
                    const aiData = aiMatchKey ? data[aiMatchKey] : null;
                    if (aiData) { return { ...row, ai_rank: aiData.rank, ai_bias: aiData.bias, ai_reason: aiData.reason }; }
                    return row;
                });
                return [...newResults];
            });
        } catch (err) { setError(err.message); } finally { setRankingLLM(false); }
    };

    const handleAnalyzeLLMRow = async (symbolToAnalyze: string) => {
        setAnalyzingLLM(symbolToAnalyze);
        try {
            const res = await fetch(`${API_BASE}/llm/analyze_row/${symbolToAnalyze}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Failed to analyze with LLM");
            setAnalysisResult({ ...data, pair: symbolToAnalyze });
        } catch (err) { setError(`Error analyzing ${symbolToAnalyze}: ${err.message}`); } finally { setAnalyzingLLM(null); }
    };

    const handleQuickTrade = (setup) => {
        onAutoTrade(setup);
        navigate('/');
    };

    const fetchLatestScan = async () => {
        try {
            const res = await fetch(`${API_BASE}/scanner/table`);
            const data = await res.json();
            if (res.ok && data.results) {
                setScannerResults(data.results);
                if (data.base_timeframe) setLastScanTF(data.base_timeframe);
            }
        } catch (err) { console.error("Error fetching latest scan:", err); }
    };

    useEffect(() => { fetchLatestScan(); }, []);

    const getScoreColor = (score) => {
        if (score >= 7.5) return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20';
        if (score >= 5.5) return 'text-blue-400 bg-blue-400/10 border-blue-400/20';
        if (score <= 3.5) return 'text-rose-400 bg-rose-400/10 border-rose-400/20';
        return 'text-slate-400 bg-slate-400/10 border-slate-400/20';
    };

    const getStatusColor = (status) => {
        if (status === 'STRONG_BULLISH' || status === 'BULLISH' || status === 'BREAKOUT') return 'text-emerald-400';
        if (status === 'STRONG_BEARISH' || status === 'BEARISH') return 'text-rose-400';
        return 'text-slate-400';
    };

    return (
        <div className="h-full flex flex-col bg-slate-900 overflow-hidden">
            <AnalysisModal data={analysisResult} onClose={() => setAnalysisResult(null)} onQuickTrade={handleQuickTrade} />
            <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center px-6 py-6 gap-6">
                <div>
                    <h1 className="text-3xl font-black text-white flex items-center gap-3 tracking-tighter">
                        <ScanSearch size={32} className="text-blue-500" /> QUANT SCANNER
                    </h1>
                    <div className="flex items-center gap-2 mt-1 group relative cursor-help">
                        <p className="text-slate-500 text-xs font-bold uppercase tracking-widest">Top 20 Opportunity Markets (USDC)</p>
                        <Info size={12} className="text-slate-600" />
                        <div className="absolute top-full left-0 mt-2 w-64 bg-slate-800 p-3 rounded-lg border border-slate-700 shadow-2xl z-50 invisible group-hover:visible text-[10px] text-slate-300 leading-relaxed font-medium">
                            <span className="text-blue-400 font-bold block mb-1 uppercase">Selection Logic:</span>
                            Ranked using a multi-factor Opportunity Score:<br/>
                            • 30% 24h Volume (&gt;1M USDC)<br/>
                            • 25% Volatility (ATR)<br/>
                            • 20% Momentum (RSI)<br/>
                            • 15% Trend Strength (ADX)<br/>
                            • 10% Recent Price Move
                        </div>
                    </div>
                </div>
                <div className="flex flex-wrap items-center gap-4">
                    <div className="relative group flex items-center">
                        <Filter size={14} className="absolute left-3 text-slate-500 group-focus-within:text-blue-400 transition-colors" />
                        <input type="text" placeholder="Search Pair..." value={filterText} onChange={(e) => setFilterText(e.target.value)} className="pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs font-black outline-none focus:border-blue-500 w-48 transition-all focus:ring-1 focus:ring-blue-500/20" />
                        {filterText && (<button onClick={() => setFilterText('')} className="absolute right-3 text-[10px] font-black text-slate-600 hover:text-slate-400">CLEAR</button>)}
                    </div>
                    <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800">
                        {timeframes.map(tf => (<button key={tf} onClick={() => setSelectedTF(tf)} className={`px-4 py-1.5 rounded-lg text-[10px] font-black transition-all ${selectedTF === tf ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}>{tf}</button>))}
                    </div>
                    <div className="flex gap-3">
                        <button onClick={handleRunScan} disabled={scanning} className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-black py-2.5 px-6 rounded-xl transition-all disabled:opacity-50 shadow-lg shadow-blue-900/20 text-xs uppercase tracking-wider">
                            <Zap size={16} className={scanning ? 'animate-spin' : ''} /> {scanning ? 'SCANNING...' : `SCAN ON ${selectedTF}`}
                        </button>
                        <button onClick={handleRankLLM} disabled={rankingLLM || scannerResults.length === 0} className="flex items-center gap-2 bg-purple-600 hover:bg-purple-500 text-white font-black py-2.5 px-6 rounded-xl transition-all disabled:opacity-50 shadow-lg shadow-purple-900/20 text-xs uppercase tracking-wider">
                            <Target size={16} className={rankingLLM ? 'animate-pulse' : ''} /> {rankingLLM ? 'RANKING...' : 'AI RANK'}
                        </button>
                    </div>
                </div>
            </header>
            {error && (<div className="mb-6 p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl flex items-center gap-3 mx-6"><div className="w-2 h-2 bg-rose-500 rounded-full animate-ping"></div><p className="text-rose-400 text-xs font-bold uppercase tracking-wider">Error: {error}</p></div>)}
            <div className="flex-1 min-h-0 bg-slate-950 rounded-3xl border border-slate-800 shadow-2xl overflow-hidden mx-6 mb-6">
                <div className="bg-slate-900/30 px-6 py-3 border-b border-slate-800 flex justify-between items-center">
                    <div className="flex items-center gap-2"><Activity size={14} className="text-blue-400" /><span className="text-[10px] text-slate-400 font-bold uppercase tracking-[0.1em]">Displaying <span className="text-blue-400">{lastScanTF}</span> analysis • Sorted by <span className="text-blue-400">{sortConfig.key}</span></span></div>
                    <span className="text-[9px] text-slate-600 font-bold uppercase tracking-widest">{processedResults.length} / {scannerResults.length} Pairs</span>
                </div>
                <div className="h-full overflow-auto">
                    <table className="w-full text-left border-collapse">
                        <thead className="sticky top-0 bg-slate-900 z-20">
                            <tr className="bg-slate-900 text-slate-500 border-b border-slate-800">
                                <ColumnHeader label="Pair" tooltip="Asset symbol compared against USDC. Top 20 are selected based on Volume, Volatility, and Momentum." sortKey="pair" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="Price" tooltip="Latest closing price from the Binance exchange." sortKey="price" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="Heatmap" tooltip="Trend Alignment: Bullish if Price > EMA20 > EMA50 > EMA200. Strong if ADX > 25." sortKey="heatmap" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="Structure" tooltip="Detects HH/HL patterns. BREAKOUT triggers if price exceeds the 20-period maximum high." sortKey="structure" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="RSI" tooltip="Relative Strength Index (14). Measures momentum speed. <30 is Oversold, >70 is Overbought." sortKey="rsi" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="MACD" tooltip="Moving Average Convergence Divergence Histogram. Positive values indicate bullish momentum shift." sortKey="macd_hist" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="ADX" tooltip="Average Directional Index (14). Measures trend strength. Values >25 indicate a strong trending market." sortKey="adx" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="Vol Ratio" tooltip="Current Volume vs 20-period Average. >1.5 indicates a significant volume spike." sortKey="volume_ratio" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="ATR Ratio" tooltip="Current Volatility (ATR) vs 20-period Average. >1.3 indicates volatility expansion." sortKey="atr_ratio" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="BB Pos" tooltip="Bollinger Band % Position. 0% is the lower band (Oversold), 100% is the upper band (Overbought)." sortKey="bb_pos" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="Score" tooltip="Quant Confluence Score (0-10). Weighted sum of all technical indicators." sortKey="score" currentSort={sortConfig} onSort={handleSort} />
                                <ColumnHeader label="AI Rank" tooltip="Independent AI ranking (1-20). The model analyzes all technicals to find the best setups." sortKey="ai_rank" currentSort={sortConfig} onSort={handleSort} />
                                <th className="py-5 px-4 text-right text-[10px] font-black uppercase tracking-[0.1em]">Action</th>
                            </tr>
                        </thead>
                        <tbody className="font-mono text-[13px]">
                            {processedResults.map((r, index) => (
                                <tr key={r.pair || index} className="border-b border-slate-800/30 hover:bg-blue-500/5 transition-all group">
                                    <td className="py-4 px-4"><div className="font-black text-white group-hover:text-blue-400 transition-colors">{r.pair}</div></td>
                                    <td className="py-4 px-4 text-slate-300">${r.price?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</td>
                                    <td className="py-4 px-4"><div className="flex flex-col gap-1.5"><span className={`text-[10px] font-black tracking-widest ${getStatusColor(r.heatmap)}`}>{r.heatmap}</span>{r.heatmap_multi && <HeatmapCell heatmap={r.heatmap_multi} />}</div></td>
                                    <td className="py-4 px-4"><span className={`px-2 py-1 rounded text-[10px] font-black border ${r.structure === 'BREAKOUT' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 animate-pulse' : 'bg-slate-800 border-slate-700 text-slate-300'}`}>{r.structure}</span></td>
                                    <td className="py-4 px-4 text-center font-bold text-slate-400">{r.rsi?.toFixed(1)}</td>
                                    <td className="py-4 px-4 text-center"><span className={`font-bold ${r.macd_hist > 0 ? 'text-emerald-500' : 'text-rose-500'}`}>{r.macd_hist > 0 ? '+' : ''}{r.macd_hist?.toFixed(2)}</span></td>
                                    <td className="py-4 px-4 text-center"><div className="flex flex-col items-center gap-1"><span className="text-slate-300 font-bold">{r.adx?.toFixed(0)}</span><div className="w-8 h-1 bg-slate-800 rounded-full overflow-hidden"><div className="h-full bg-blue-500" style={{ width: `${Math.min(100, (r.adx / 50) * 100)}%` }}></div></div></div></td>
                                    <td className="py-4 px-4 text-center"><span className={`font-black ${r.volume_ratio > 1.5 ? 'text-blue-400' : 'text-slate-500'}`}>{r.volume_ratio?.toFixed(1)}x</span></td>
                                    <td className="py-4 px-4 text-center"><span className={`font-black ${r.atr_ratio > 1.3 ? 'text-orange-400' : 'text-slate-500'}`}>{r.atr_ratio?.toFixed(1)}x</span></td>
                                    <td className="py-4 px-4 text-center"><div className="flex flex-col items-center gap-1"><span className="text-[10px] text-slate-500 font-bold">{(r.bb_pos * 100).toFixed(0)}%</span><div className="w-12 h-1.5 bg-slate-800 rounded-full relative"><div className="absolute top-0 h-full w-1 bg-white rounded-full transition-all" style={{ left: `${Math.max(0, Math.min(100, r.bb_pos * 100))}%` }}></div></div></div></td>
                                    <td className="py-4 px-4 text-center"><div className={`inline-block px-3 py-1 rounded-full font-black text-xs border ${getScoreColor(r.score)}`}>{r.score?.toFixed(1)}</div></td>
                                    <td className="py-4 px-4 text-center">{r.ai_rank && r.ai_rank !== 'N/A' ? (<div className="flex flex-col items-center"><span className="text-purple-400 font-black text-lg">#{r.ai_rank}</span><span className={`text-[9px] font-bold uppercase tracking-tighter ${r.ai_bias?.toLowerCase() === 'bullish' ? 'text-emerald-400' : 'text-rose-400'}`}>{r.ai_bias}</span></div>) : <span className="text-slate-700">--</span>}</td>
                                    <td className="py-4 px-4 text-right">
                                        <button onClick={() => handleAnalyzeLLMRow(r.pair)} disabled={analyzingLLM === r.pair} className="group/btn relative p-2 bg-slate-800 hover:bg-blue-600 rounded-lg transition-all disabled:opacity-30" title="AI Deep Analysis">
                                            {analyzingLLM === r.pair ? <Zap size={16} className="text-blue-400 animate-spin" /> : <Activity size={16} className="text-blue-400 group-hover/btn:text-white" />}
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default ScannerView;
