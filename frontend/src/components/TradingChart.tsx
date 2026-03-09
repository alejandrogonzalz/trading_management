import { useEffect, useRef } from 'react';
import { createChart, ColorType } from 'lightweight-charts';

const TradingChart = () => {
  const chartContainerRef = useRef();

  useEffect(() => {
    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    };

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 440,
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#4ade80',
      downColor: '#f87171',
      borderVisible: false,
      wickUpColor: '#4ade80',
      wickDownColor: '#f87171',
    });

    // Initial dummy data
    candlestickSeries.setData([
      { time: '2024-03-01', open: 62000, high: 64000, low: 61000, close: 63000 },
      { time: '2024-03-02', open: 63000, high: 65000, low: 62000, close: 64000 },
      { time: '2024-03-03', open: 64000, high: 68000, low: 63000, close: 67000 },
      { time: '2024-03-04', open: 67000, high: 69000, low: 66000, close: 68000 },
    ]);

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  return <div ref={chartContainerRef} className="w-full h-full" />;
};

export default TradingChart;
