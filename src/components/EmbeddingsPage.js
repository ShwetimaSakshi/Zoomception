import React, { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

const EmbeddingsPage = () => {
  const plotRef = useRef();
  const tooltipRef = useRef();

  useEffect(() => {
    const data = [
      {
        x: [1, 2],
        y: [2, 3],
        z: [3, 1],
        mode: "markers",
        type: "scatter3d",
        text: ["Dog Text", "Cat Text"],
        customdata: [
          ["Dog Text", "/dog.jpg", "text"],
          ["Cat Text", "/cat.jpg", "text"],
        ],
        hoverinfo: "text",
        marker: {
          size: 10,
          symbol: "circle",
          color: "blue",
        },
        name: "Text Markers",
      },
     
      {
        x: [1.5, 2.5],
        y: [2.5, 3.5],
        z: [3.5, 1.5],
        mode: "markers",
        type: "scatter3d",
        text: ["Dog Image", "Cat Image"],
        customdata: [
          ["Dog Text", "/dog.jpg", "image"],
          ["Cat Text", "/cat.jpg", "image"],
        ],
        hoverinfo: "text",
        marker: {
          size: 10,
          symbol: "square",
          color: "green",
        },
        name: "Image Markers",
      },
    ];

    const layout = {
      height: 600,
      margin: {
        l: 50, 
        r: 50, 
        b: 100,
        t: 50, 
      },
      legend: {
        orientation: "h",
        y: -0.2,          
        yanchor: "bottom",
        x: 0.5,           
        xanchor: "center",
      },
    };

    Plotly.newPlot(plotRef.current, data, layout);

    const plotDiv = plotRef.current;
    const tooltip = tooltipRef.current;

    plotDiv.on("plotly_hover", (data) => {
      const point = data.points[0];
      const text = point.customdata[0];
      const img = point.customdata[1];

      tooltip.innerHTML = `
        <div style="background: white; border: 1px solid #ccc; padding: 5px;">
          <div>${text}</div>
          <img src="${img}" width="100" />
        </div>
      `;

      tooltip.style.display = "block";
    });

    plotDiv.on("plotly_unhover", () => {
      tooltip.style.display = "none";
    });

    const handleMouseMove = (e) => {
      tooltip.style.left = `${e.clientX + 10}px`;
      tooltip.style.top = `${e.clientY + 10}px`;
    };

    window.addEventListener("mousemove", handleMouseMove);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  return (
    <div style={{ position: "relative" }}>
      <div ref={plotRef} />
      <div
        ref={tooltipRef}
        style={{
          position: "fixed",
          display: "none",
          pointerEvents: "none",
          zIndex: 10,
        }}
      />
    </div>
  );
};

export default EmbeddingsPage;
