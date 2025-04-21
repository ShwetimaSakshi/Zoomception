import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import data from "../assets/clip-data.json";
import { Box, Paper, Typography } from "@mui/material";

function ClipModelViz() {
  const svgRef = useRef();
  const [hoverInfo, setHoverInfo] = useState(null);

  useEffect(() => {
    // Get screen width and height with a margin
    const margin = { top: 50, right: 50, bottom: 50, left: 50 };
    const width = (window.innerWidth * 0.6) - margin.left - margin.right; // Half width for the visualization
    const height = window.innerHeight - margin.top - margin.bottom;

    // Clear previous render
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`) // Add viewBox for better scaling
      .attr("preserveAspectRatio", "xMidYMid meet"); // This helps center the content

    const g = svg.append("g").attr("transform", `translate(${width/2}, ${height/2})`);

    // Create hierarchy
    const root = d3.hierarchy(data);
    
    // Track navigation history
    let currentNode = root;
    let nodeHistory = [];
    // Store active timers to clean up
    let activeTimers = [];

    // Initial display - just the root
    update(root);

    function update(node) {
      // Clear the previous nodes and stop any active animations
      g.selectAll("*").remove();
      activeTimers.forEach(timer => timer.stop && timer.stop());
      activeTimers = [];
      
      // Create a single node for the current node
      if (!node.parent) {
        // This is the root or a top-level node
        const rootNode = g.append("g")
          .attr("class", "node")
          .attr("transform", `translate(0,0)`)
          .style("opacity", 0) // Start invisible
          .on("click", (event) => {
            if (node.children && node.children.length > 0) {
              nodeHistory.push(node);
              currentNode = node;
              showChildren(node);
            }
          })
          .on("mouseover", () => {
            // Set hover info when mouse is over the node
            setHoverInfo({
              name: node.data.name,
              description: node.data.description || "No description available",
              properties: node.data.properties || {}
            });
          })
          .on("mouseout", () => {
            // Clear hover info when mouse leaves
            setHoverInfo(null);
          });
          
        // Get text width for the box
        const getTextWidth = (text) => {
          const canvas = document.createElement("canvas");
          const context = canvas.getContext("2d");
          context.font = "14px 'Roboto', sans-serif"; // Updated font
          return context.measureText(text).width + 80;
        };
        
        const boxWidth = getTextWidth(node.data.name);
        
        // Draw box with enhanced styling
        rootNode.append("rect")
          .attr("width", boxWidth)
          .attr("height", 40)
          .attr("x", -boxWidth / 2)
          .attr("y", -20)
          .attr("rx", 10)
          .attr("fill", "#003366")
          .attr("stroke", "#4a90e2") // Use secondary color for border
          .attr("stroke-width", 2); // Slightly thicker border
          
        // Add label with improved font
        rootNode.append("text")
          .attr("dy", 5)
          .attr("text-anchor", "middle")
          .attr("fill", "white")
          .attr("font-family", "'Roboto', sans-serif") // Specify font family
          .attr("font-weight", 500) // Medium weight
          .attr("font-size", "14px") // Slightly larger font
          .text(node.data.name)
          .style("cursor", "pointer");
          
        // Add indicator if it has children
        if (node.children && node.children.length > 0 && node.data.name !== "CLIP Model") {
          rootNode.append("text")
            .attr("dy", 5)
            .attr("dx", boxWidth / 2 + 5)
            .attr("text-anchor", "start")
            .attr("fill", "white")
            .attr("font-family", "'Roboto', sans-serif")
            .attr("font-weight", 500)
            .attr("font-size", "14px")
            .text("+")
            .style("cursor", "pointer");
            
          // Add animation with reduced scale range for nodes with children
          rootNode.transition()
            .duration(750)
            .style("opacity", 1)
            .attrTween("transform", function() {
              return function(t) {
                // Reduced scale range (0.85 to 1.0 instead of 0.7 to 1.0)
                const scale = d3.easeCubic(t) * 0.05 + 0.9; // Smaller range with smoother easing
                return `translate(0,0) scale(${scale})`;
              };
            });
            
          // After initial animation, add subtle continuous pulse
          setTimeout(() => {
            const startTime = Date.now();
            const timer = d3.timer(function() {
              const elapsed = Date.now() - startTime;
              // Very subtle pulse (only 1.5% variation)
              const scale = 1 + 0.015 * Math.sin(elapsed / 500);
              
              rootNode.attr("transform", `translate(0,0) scale(${scale})`);
              
              // Return true to stop the timer if node is no longer in the DOM
              return !rootNode.node();
            });
            
            activeTimers.push(timer);
          }, 750);
        } else {
          // Simple fade-in for nodes without children
          rootNode.transition()
            .duration(500)
            .style("opacity", 1);
        }
      }
    }
    
    function showChildren(parentNode) {
      const duration = 750;
      
      // Clear the previous nodes and stop any active animations
      g.selectAll("*").remove();
      activeTimers.forEach(timer => timer.stop && timer.stop());
      activeTimers = [];
      
      // Create a blur filter with reduced intensity
      const defs = svg.append("defs");
      const filter = defs.append("filter")
        .attr("id", "blur-effect")
        .attr("x", "-50%")
        .attr("y", "-50%")
        .attr("width", "200%")
        .attr("height", "200%");
        
      filter.append("feGaussianBlur")
        .attr("in", "SourceGraphic")
        .attr("stdDeviation", "1"); // Reduced blur intensity
      
      // Helper function for text width
      const getTextWidth = (text) => {
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");
        context.font = "14px 'Roboto', sans-serif"; // Updated font
        return context.measureText(text).width + 80;
      };
      
      // Draw the parent node with reduced blur
      const parentWidth = getTextWidth(parentNode.data.name);
      const parentGroup = g.append("g")
        .attr("class", "parent-node")
        .attr("transform", "translate(0,0)")
        .style("opacity", 0.8) // Slightly increased opacity
        .style("filter", "url(#blur-effect)")
        .style("cursor", "pointer") // Make it look clickable
        .on("click", () => {
          if (nodeHistory.length > 0) {
            currentNode = nodeHistory.pop();
            if (nodeHistory.length > 0) {
              showChildren(nodeHistory[nodeHistory.length - 1]);
            } else {
              update(root);
            }
          }
        })
        .on("mouseover", () => {
          setHoverInfo({
            name: parentNode.data.name,
            description: parentNode.data.description || "No description available",
            properties: parentNode.data.properties || {}
          });
        })
        .on("mouseout", () => {
          setHoverInfo(null);
        });
        
      parentGroup.append("rect")
        .attr("width", parentWidth)
        .attr("height", 40)
        .attr("x", -parentWidth / 2)
        .attr("y", -20)
        .attr("rx", 10)
        .attr("fill", "#003366")
        .attr("stroke", "#4a90e2") // Updated border color
        .attr("stroke-width", 2); // Thicker border
        
      parentGroup.append("text")
        .attr("dy", 5)
        .attr("text-anchor", "middle")
        .attr("fill", "white")
        .attr("font-family", "'Roboto', sans-serif") // Specify font family
        .attr("font-weight", 500) // Medium weight
        .attr("font-size", "14px") // Slightly larger font
        .text(parentNode.data.name);
      
      // Calculate positions for the children in a circle
      const children = parentNode.children || [];
      const radius = Math.min(width, height) * 0.3;
      const angleStep = (2 * Math.PI) / children.length;
      
      // Store all node positions to use for path calculation
      const nodePositions = children.map((child, index) => {
        const angle = index * angleStep;
        return {
          node: child,
          x: radius * Math.cos(angle),
          y: radius * Math.sin(angle),
          width: getTextWidth(child.data.name),
          height: 40,
          angle: angle
        };
      });
      
      // Now draw the nodes (on top of the lines)
      nodePositions.forEach(nodePos => {
        const childNode = g.append("g")
          .attr("class", "node")
          .attr("transform", `translate(0,0)`) // Start at center
          .style("transform-origin", "center")
          .style("opacity", 0) // Start invisible
          .on("click", (event) => {
            if (nodePos.node.children && nodePos.node.children.length > 0) {
              nodeHistory.push(currentNode);
              currentNode = nodePos.node;
              showChildren(nodePos.node);
            }
          })
          .on("mouseover", () => {
            setHoverInfo({
              name: nodePos.node.data.name,
              description: nodePos.node.data.description || "No description available",
              properties: nodePos.node.data.properties || {}
            });
          })
          .on("mouseout", () => {
            setHoverInfo(null);
          });
          
        // First transition to position with fade-in
        childNode.transition()
          .duration(duration)
          .attr("transform", `translate(${nodePos.x},${nodePos.y})`)
          .style("opacity", 1);
        
        // Draw box with enhanced styling
        childNode.append("rect")
          .attr("width", nodePos.width)
          .attr("height", 40)
          .attr("x", -nodePos.width / 2)
          .attr("y", -20)
          .attr("rx", 10)
          .attr("fill", "#003366")
          .attr("stroke", "#4a90e2") // Use secondary color for border
          .attr("stroke-width", 2); // Slightly thicker border
          
        // Add label with improved font
        childNode.append("text")
          .attr("dy", 5)
          .attr("text-anchor", "middle")
          .attr("fill", "white")
          .attr("font-family", "'Roboto', sans-serif") // Specify font family
          .attr("font-weight", 500) // Medium weight
          .attr("font-size", "14px") // Slightly larger font
          .text(nodePos.node.data.name)
          .style("cursor", "pointer");
          
        // Add indicator if it has children
        if (nodePos.node.children && nodePos.node.children.length > 0) {
          childNode.append("text")
            .attr("dy", 5)
            .attr("dx", nodePos.width / 2 + 5)
            .attr("text-anchor", "start")
            .attr("fill", "white")
            .attr("font-family", "'Roboto', sans-serif")
            .attr("font-weight", 500)
            .attr("font-size", "14px")
            .text("+")
            .style("cursor", "pointer");
            
          // Add subtle pulse animation for nodes with children
          // Wait for initial positioning to complete
          setTimeout(() => {
            const startTime = Date.now();
            const timer = d3.timer(function() {
              const elapsed = Date.now() - startTime;
              // Very subtle pulse (only 1.5% variation)
              const scale = 1 + 0.015 * Math.sin(elapsed / 500);
              const baseTransform = `translate(${nodePos.x},${nodePos.y})`;
              
              childNode.attr("transform", `${baseTransform} scale(${scale})`);
              
              // Return true to stop the timer if node is no longer in the DOM
              return !childNode.node();
            });
            
            activeTimers.push(timer);
          }, duration);
        }
      });
    }
    
    // Add navigation buttons
    const buttonGroup = svg.append("g")
      .attr("transform", "translate(50, 30)");
    
    // Create a more attractive Reset button
    const resetButton = buttonGroup.append("g")
      .attr("class", "reset-button")
      .style("cursor", "pointer")
      .on("click", () => {
        currentNode = root;
        nodeHistory = [];
        update(root);
      });

    // Add button background with rounded corners
    resetButton.append("rect")
      .attr("x", -40)
      .attr("y", -15)
      .attr("width", 120)
      .attr("height", 30)
      .attr("rx", 8)  // Rounded corners
      .attr("ry", 8)
      .attr("fill", "#003366")  // Match your node color
      .attr("stroke", "#4a90e2") // Updated border color
      .attr("stroke-width", 2); // Thicker border

    // Add button text
    resetButton.append("text")
      .attr("x", 20)
      .attr("y", 5)
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .attr("font-family", "'Roboto', sans-serif") // Specify font family
      .attr("font-weight", 500) // Medium weight
      .attr("font-size", "14px") // Slightly larger font
      .text("Reset to CLIP");
    
    // Clean up function to stop all animations when component unmounts
    return () => {
      activeTimers.forEach(timer => timer.stop && timer.stop());
    };
    
  }, []);

  return (
    <Box sx={{ 
      display: 'flex', 
      width: '100%', 
      border: '1px solid #ccc', 
      borderRadius: '12px',
      overflow: 'hidden',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
      backgroundColor: '#fff'
    }}>
      {/* Left panel - Visualization */}
      <Box sx={{ width: '70%', padding: '24px', borderRight: '1px solid #eaeaea' }}>
        <Typography 
          variant="h5" 
          sx={{ 
            textAlign: "center", 
            mb: 3,
            fontWeight: 600,
            color: '#003366',
            letterSpacing: '0.5px'
          }}
        >
          CLIP Model Architecture
        </Typography>
        <svg ref={svgRef}></svg>
      </Box>
      
      {/* Right panel - Node Information */}
      <Box sx={{ width: '30%', padding: '24px', backgroundColor: '#f8f9fa' }}>
        <Typography 
          variant="h5" 
          sx={{ 
            textAlign: "center", 
            mb: 3,
            fontWeight: 600,
            color: '#003366',
            letterSpacing: '0.5px'
          }}
        >
          Component Details
        </Typography>
        {hoverInfo ? (
          <Paper 
            elevation={0} 
            sx={{ 
              p: 3, 
              height: '80%', 
              overflow: 'auto',
              border: '1px solid #eaeaea',
              borderRadius: '8px'
            }}
          >
            <Typography 
              variant="h6" 
              gutterBottom
              sx={{ 
                color: '#003366',
                fontWeight: 600,
                pb: 1,
                borderBottom: '2px solid #4a90e2'
              }}
            >
              {hoverInfo.name}
            </Typography>
            <Typography 
              variant="body1" 
              paragraph
              sx={{ 
                fontSize: '0.95rem',
                lineHeight: 1.6,
                color: 'text.primary'
              }}
            >
              {hoverInfo.description}
            </Typography>
            {Object.keys(hoverInfo.properties).length > 0 && (
              <>
                <Typography 
                  variant="subtitle1" 
                  sx={{ 
                    mt: 2, 
                    mb: 1,
                    fontWeight: 600,
                    color: '#003366'
                  }}
                >
                  Properties:
                </Typography>
                {Object.entries(hoverInfo.properties).map(([key, value]) => (
                  <Typography 
                    key={key} 
                    variant="body2"
                    sx={{ 
                      mb: 1,
                      fontSize: '0.9rem',
                      '& strong': {
                        fontWeight: 600,
                        color: '#4a90e2'
                      }
                    }}
                  >
                    <strong>{key}:</strong> {value}
                  </Typography>
                ))}
              </>
            )}
          </Paper>
        ) : (
          <Paper 
            elevation={0} 
            sx={{ 
              p: 3, 
              height: '80%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              border: '1px solid #eaeaea',
              borderRadius: '8px',
              backgroundColor: 'rgba(0, 51, 102, 0.02)'
            }}
          >
            <Typography 
              variant="body1" 
              color="text.secondary"
              sx={{ 
                fontStyle: 'italic',
                textAlign: 'center'
              }}
            >
              Hover over a node in the visualization to view its detailed information
            </Typography>
          </Paper>
        )}
      </Box>
    </Box>
  );
}

export default ClipModelViz;
