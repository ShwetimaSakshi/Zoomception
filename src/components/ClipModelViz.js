import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import data from "../assets/clip-data.json";

function ClipModelViz() {
  const svgRef = useRef();

  useEffect(() => {
   // Get screen width and height with a margin
   const margin = { top: 50, right: 50, bottom: 50, left: 50 };
   const width = window.innerWidth - margin.left - margin.right;
   const height = window.innerHeight - margin.top - margin.bottom;

    // Clear previous render
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    const g = svg.append("g").attr("transform", `translate(150, 50)`);

    // Create hierarchy
    const root = d3.hierarchy(data);
    root.x0 = height / 2;
    root.y0 = 0;

    const treeLayout = d3.tree().size([height - 100, width - 100]);

    // Collapse all children initially
    root.children.forEach(collapse);

    let i = 0; // ✅ FIXED: declare before use

    update(root);

    function collapse(d) {
      if (d.children) {
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
      }
    }

    function update(source) {
      const duration = 750;
      const treeData = treeLayout(root);

      const nodes = treeData.descendants();
      const links = treeData.links();

      nodes.forEach((d) => {
        d.y = d.depth * 180;
      });

      // Join nodes
      const node = g.selectAll("g.node").data(nodes, (d) => d.id || (d.id = ++i));

      // Enter new nodes
      const nodeEnter = node
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", () => `translate(${source.y0},${source.x0})`)
        .on("click", (_, d) => {
          if (d.children) {
            d._children = d.children;
            d.children = null;
          } else {
            d.children = d._children;
            d._children = null;
          }
          update(d);
        });

      // Dynamically adjust width and height based on label length
      const getTextWidth = (text) => {
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");
        context.font = "12px sans-serif"; // Adjust based on your text style
        return context.measureText(text).width + 80; // Adding padding
      };

      // Draw boxes
      nodeEnter
        .append("rect")
        .attr("width", (d) => getTextWidth(d.data.name))
        .attr("height", 40)
        .attr("x", (d) => -getTextWidth(d.data.name) / 2)
        .attr("y", -20)
        .attr("rx", 10)
        .attr("fill", "#003366")
        .attr("stroke", "#ccc")
        .attr("stroke-width", 1.5);

      // Add labels
      nodeEnter
        .append("text")
        .attr("dy", 5)
        .attr("text-anchor", "middle")
        .attr("fill", "white")
        .text((d) => d.data.name)
        .style("cursor", "pointer");

      // Update positions
      const nodeUpdate = nodeEnter
        .merge(node)
        .transition()
        .duration(duration)
        .attr("transform", (d) => `translate(${d.y},${d.x})`);

      // Remove old nodes
      const nodeExit = node
        .exit()
        .transition()
        .duration(duration)
        .attr("transform", () => `translate(${source.y},${source.x})`)
        .remove();

      nodeExit.select("rect").attr("fill-opacity", 1e-6);
      nodeExit.select("text").attr("fill-opacity", 1e-6);

      // Join links
      const link = g.selectAll("path.link").data(links, (d) => d.target.id);

      const diagonal = d3
        .linkHorizontal()
        .x((d) => d.y)
        .y((d) => d.x);

      const linkEnter = link
        .enter()
        .insert("path", "g")
        .attr("class", "link")
        .attr("d", () => {
          const o = { x: source.x0, y: source.y0 };
          return diagonal({ source: o, target: o });
        })
        .attr("fill", "none")
        .attr("stroke", "#aaa")
        .attr("stroke-width", 2);

      linkEnter.merge(link).transition().duration(duration).attr("d", diagonal);

      link
        .exit()
        .transition()
        .duration(duration)
        .attr("d", () => {
          const o = { x: source.x, y: source.y };
          return diagonal({ source: o, target: o });
        })
        .remove();

      // Save old positions
      nodes.forEach((d) => {
        d.x0 = d.x;
        d.y0 = d.y;
      });
    }
  }, []);

  return (
    <div>
      <h2 style={{ textAlign: "center" }}>CLIP Model Architecture</h2>
      <svg ref={svgRef}></svg>
    </div>
  );
}

export default ClipModelViz;
