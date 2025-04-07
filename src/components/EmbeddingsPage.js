import React, { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import { Typography } from "@mui/material";

const EmbeddingsPage = () => {
  const plotRef = useRef();
  const tooltipRef = useRef();

  useEffect(() => {
    fetch("/data/embeddings_3d_coordinates.json")
      .then((response) => response.json())
      .then((jsonData) => {
        const images = jsonData.images;
        console.log("Loaded JSON:", jsonData);

        const labelX = [], labelY = [], labelZ = [];
        const imageX = [], imageY = [], imageZ = [];
        const linesX = [], linesY = [], linesZ = [];
        
        const offset = 20;
        
        images.forEach((item) => {
          const [x, y, z] = item.coordinates;
          const label = item.metadata.label;
          const url = item.metadata.url;
        
          // Original coordinates for label
          labelX.push(x);
          labelY.push(y);
          labelZ.push(z);
        
          // Image marker: offset only in x, same y and z
          const imageXCoord = x + offset;
          imageX.push(imageXCoord);
          imageY.push(y);
          imageZ.push(z);
        
          // Add connecting line between label and image
          linesX.push(x, imageXCoord, null); // null separates segments
          linesY.push(y, y, null);
          linesZ.push(z, z, null);
        });
        
        // Plot data array
        const plotData = [
          {
            x: labelX,
            y: labelY,
            z: labelZ,
            mode: "markers",
            type: "scatter3d",
            text: images.map((item) => item.metadata.label),
            hoverinfo: "text",
            marker: {
              size: 5,
              color: "green",
            },
            name: "Labels",
          },
          {
            x: imageX,
            y: imageY,
            z: imageZ,
            mode: "markers",
            type: "scatter3d",
            customdata: images.map((item) => [item.metadata.label, item.metadata.url]),
            // text: images.map((item) => item.metadata.label),
            hoverinfo: "text",
            marker: {
              size: 5,
              color: "orange",
            },
            name: "Images",
          },
          {
            x: linesX,
            y: linesY,
            z: linesZ,
            mode: "lines",
            type: "scatter3d",
            line: {
              color: "black",
              width: 2,
            },
            hoverinfo: "none",
            name: "Connections",
            showlegend: false,
          },
        ];
  
        const layout = {
          height: 600,
          margin: { l: 50, r: 50, b: 100, t: 50 },
          legend: {
            orientation: "h",
            y: -0.2,
            yanchor: "bottom",
            x: 0.5,
            xanchor: "center",
          },
        };
  
        Plotly.newPlot(plotRef.current, plotData, layout);
  
        const plotDiv = plotRef.current;
        const tooltip = tooltipRef.current;
  
        plotDiv.on("plotly_hover", (data) => {
          const point = data.points[0];
          
          const customdata = point.customdata;
          if (!customdata) return; 

          const label = point.customdata[0];
          const imageUrl = point.customdata[1];
  
          tooltip.innerHTML = `
            <div style="background: white; border: 1px solid #ccc; padding: 5px;">
              <div style="font-size: 12px; margin-bottom: 5px;">${label}</div>
              <img src="${imageUrl}" width="120" />
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
      })
      .catch((error) => {
        console.error("Error loading embeddings JSON:", error);
      });
  }, []);


  return (
    <div style={{ position: "relative" }}>
      <Typography variant="body1" paragraph>
      Hover on data points to view the embedding mappings.
      </Typography>
      
  
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
