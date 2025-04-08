import React, { useEffect, useRef, useState, useCallback } from "react";
import Plotly from "plotly.js-dist-min";
import { Typography } from "@mui/material";
import Plot from "react-plotly.js";
import "../App.css";
import { debounce } from 'lodash';

const EmbeddingsPage = () => {
  const plotRef = useRef();
  const [hoverDetails, setHoverDetails] = useState(null);
  const [matrixDetails, setMatrixDetails] = useState(null);

  const [similarityMatrix, setSimilarityMatrix] = useState([]);
  const [imageIndices, setImageIndices] = useState([]);
  const [textIndices, setTextIndices] = useState([]);
  const [texts, setTexts] = useState([]);
  const [imageInfo, setImageInfo] = useState([]);
  const [hoverText, setHoverText] = useState([]);

  useEffect(() => {
    fetch("/data/embeddings_3d_coordinates.json")
      .then((response) => response.json())
      .then((jsonData) => {
        const images = jsonData.images;

        const labelX = [], labelY = [], labelZ = [];
        const imageX = [], imageY = [], imageZ = [];
        const linesX = [], linesY = [], linesZ = [];

        const offset = 20;

        const customDataArray = images.map((item) => [item.metadata.label, item.metadata.url]);

        images.forEach((item) => {
          const [x, y, z] = item.coordinates;
          const imageXCoord = x + offset;

          labelX.push(x);
          labelY.push(y);
          labelZ.push(z);

          imageX.push(imageXCoord);
          imageY.push(y);
          imageZ.push(z);

          linesX.push(x, imageXCoord, null);
          linesY.push(y, y, null);
          linesZ.push(z, z, null);
        });

        const plotData = [
          {
            x: labelX,
            y: labelY,
            z: labelZ,
            mode: "markers",
            type: "scatter3d",
            text: images.map((item) => item.metadata.label),
            customdata: customDataArray,
            hoverinfo: "text",
            marker: { size: 5, color: "green" },
            name: "Labels",
          },
          {
            x: imageX,
            y: imageY,
            z: imageZ,
            mode: "markers",
            type: "scatter3d",
            text: images.map((item) => [item.metadata.label, item.metadata.url]),
            customdata: customDataArray,
            marker: { size: 5, color: "orange" },
            name: "Images",
            hoverinfo: "text",
          },
          {
            x: linesX,
            y: linesY,
            z: linesZ,
            mode: "lines",
            type: "scatter3d",
            line: { color: "black", width: 2 },
            hoverinfo: "none",
            showlegend: false,
          },
        ];

        const layout = {
          height: 500,
          margin: { l: 50, r: 50, b: 50, t: 50 },
        };

        Plotly.newPlot(plotRef.current, plotData, layout);

        const plotDiv = plotRef.current;

        plotDiv.on("plotly_hover", (data) => {
          const point = data.points[0];
          const customdata = point.customdata;
          if (!customdata) {
            setHoverDetails(null);
            return;
          }

          const [label, imageUrl] = customdata;
          setHoverDetails({ label, imageUrl, type: "3D Embedding" });
        });

        plotDiv.on("plotly_unhover", () => {
          setHoverDetails(null);
        });
      })
      .catch((error) => {
        console.error("Error loading embeddings JSON:", error);
      });
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const simMatrixRes = await fetch("/data/similarity_matrix.json");
        const indexMapRes = await fetch("/data/index_mapping.json");

        const simMatrix = await simMatrixRes.json();
        const indexMap = await indexMapRes.json();

        const { image_indices, text_indices, texts, image_info } = indexMap;

        const hover = image_indices.map((_, i) =>
          text_indices.map((_, j) => {
            const img = image_info[i];
            const txt = texts[j];
            const score = simMatrix[i][j];
            return {
              label: img?.label,
              imageUrl: img?.url,
              text: txt,
              score: score,
            };
          })
        );

        setSimilarityMatrix(simMatrix);
        setImageIndices(image_indices);
        setTextIndices(text_indices);
        setTexts(texts);
        setImageInfo(image_info);
        setHoverText(hover);
      } catch (err) {
        console.error("Failed to fetch similarity or index data:", err);
      }
    };

    fetchData();
  }, []);

  const handleHover = useCallback(
    debounce((event) => {
      const point = event.points[0];
      const row = imageIndices.indexOf(parseInt(point.y.replace("Image ", "")));
      const col = textIndices.indexOf(parseInt(point.x.replace("Text ", "")));

      if (row >= 0 && col >= 0 && hoverText[row]?.[col]) {
        const details = hoverText[row][col];
        setMatrixDetails({
          label: details.label,
          imageUrl: details.imageUrl,
          text: details.text,
          score: details.score,
        });
      } else {
        setMatrixDetails(null);
      }
    }, 200), // 200ms debounce delay
    [imageIndices, textIndices, hoverText] // Dependency list
  );


  return (
    <div className="container">
      <div className="section">
        <div className="split-panel">
          <Typography variant="h5" sx={{ p: 1 , fontWeight: 600}}>
            Hover on data points to view the embedding mappings.
          </Typography>
          <div ref={plotRef} />
        </div>

        <div className="details-panel">
          <Typography variant="h6">3D Plot Details</Typography>
          {hoverDetails ? (
            <div>
              <Typography variant="body2"><b>Label:</b> {hoverDetails.label}</Typography>
              <img
                src={hoverDetails.imageUrl}
                alt={hoverDetails.label}
                width="100%"
                className="panel-img"
              />
            </div>
          ) : (
            <Typography variant="body2" color="textSecondary">
              Hover over a 3D point to see details here.
            </Typography>
          )}
        </div>
      </div>

      <div className="section">
        <div className="split-panel">
          <Typography variant="h5" sx={{ p: 1 , fontWeight: 600}}>
            Image-Text Similarity Matrix
          </Typography>
          {similarityMatrix.length > 0 && hoverText.length > 0 ? (
            <Plot
              data={[
                {
                  z: similarityMatrix,
                  x: textIndices.map((i) => `Text ${i}`),
                  y: imageIndices.map((i) => `Image ${i}`),
                  type: "heatmap",
                  colorscale: "YlGnBu",
                  hoverinfo: "none",
                  text: similarityMatrix.map((row, i) =>
                    row.map((score, j) => {
                      const info = hoverText[i]?.[j];
                      return info
                        ? `Label: ${info.label}<br>Text: ${info.text}<br>Score: ${score.toFixed(3)}`
                        : `Score: ${score.toFixed(3)}`;
                    })
                  ),
                },
              ]}
              layout={{
                height: 500,
                margin: { l: 100, r: 50, b: 100, t: 50 },
                xaxis: {
                  title: {
                    text: "Text", 
                    font: { size: 16, color: 'black' }, 
                  },
                  showticklabels: false, 
                  ticks: '',
                },
                yaxis: {
                  title: {
                    text: "Images", 
                    font: { size: 16, color: 'black' }, 
                  },
                  showticklabels: false, 
                  ticks: '', 
                },
                title: "Image-Text Similarity Matrix",  
              }}
              onHover={handleHover}
              onUnhover={() => setMatrixDetails(null)}
            />
          ) : (
            <p>Loading matrix...</p>
          )}
        </div>

        <div className="details-panel">
          <Typography variant="h6">Similarity Details</Typography>
          {matrixDetails ? (
            <div>
              <Typography variant="body2"><b>Text:</b> {matrixDetails.text}</Typography>
              <Typography variant="body2"><b>Label:</b> {matrixDetails.label}</Typography>
              <Typography variant="body2"><b>Similarity:</b> {matrixDetails.score.toFixed(3)}</Typography>
              <img
                src={matrixDetails.imageUrl}
                alt={matrixDetails.label}
                width="100%"
                className="panel-img"
              />
            </div>
          ) : (
            <Typography variant="body2" color="textSecondary">
              Hover over matrix to see details here.
            </Typography>
          )}
        </div>
      </div>
    </div>
  );
};

export default EmbeddingsPage;
