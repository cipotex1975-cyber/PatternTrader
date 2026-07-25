from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.market.candles.models import Candle
from app.patterns.base_pattern import PatternResult


class ChartGenerator:
    def __init__(self) -> None:
        self._default_layout = {
            "template": "plotly_dark",
            "height": 600,
            "width": 1000,
        }

    def create_candlestick_chart(
        self,
        candles: list[Candle],
        title: str = "Price Chart",
        patterns: list[PatternResult] | None = None,
    ) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        fig.add_trace(
            go.Candlestick(
                x=[c.data.timestamp for c in candles],
                open=[c.data.open for c in candles],
                high=[c.data.high for c in candles],
                low=[c.data.low for c in candles],
                close=[c.data.close for c in candles],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=[c.data.timestamp for c in candles],
                y=[c.data.volume for c in candles],
                name="Volume",
                marker_color="rgba(0, 150, 255, 0.3)",
            ),
            row=2,
            col=1,
        )

        if patterns:
            self._add_pattern_annotations(fig, patterns)

        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            **self._default_layout,
        )

        return fig

    def _add_pattern_annotations(
        self, fig: go.Figure, patterns: list[PatternResult]
    ) -> None:
        for pattern in patterns:
            if pattern.key_levels:
                for level_name, level_price in pattern.key_levels.items():
                    fig.add_hline(
                        y=level_price,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"{level_name}: {level_price:,.2f}",
                        row=1,
                        col=1,
                    )

    def create_equity_curve(
        self,
        equity_curve: list[dict],
        title: str = "Equity Curve",
    ) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=[e["timestamp"] for e in equity_curve],
                y=[e["equity"] for e in equity_curve],
                mode="lines",
                name="Equity",
                line=dict(color="green", width=2),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Equity",
            **self._default_layout,
        )

        return fig

    def create_score_gauge(self, score: float, title: str = "Score") -> go.Figure:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": title},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 60], "color": "red"},
                        {"range": [60, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            )
        )

        fig.update_layout(**self._default_layout)
        return fig
