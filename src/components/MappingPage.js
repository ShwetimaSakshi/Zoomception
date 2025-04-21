import React, { useState, useEffect } from "react";
import {
  Box,
  Select,
  MenuItem,
  Typography,
  FormControl,
  InputLabel,
  Card,
  CardMedia,
  Stack,
  CircularProgress,
  Paper,
  IconButton,
} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search"; // Magnifying glass
import VisibilityIcon from "@mui/icons-material/Visibility"; // Eye Icon

function MappingPage() {
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [selectedTextIndex, setSelectedTextIndex] = useState(0);
  const [attentionMaps, setAttentionMaps] = useState([]);
  const [similarityMatrix, setSimilarityMatrix] = useState([]);
  const [indexMap, setIndexMap] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch the attention results, similarity matrix, and index mapping
    Promise.all([
      fetch("/data/attention_results.json"),
      fetch("/data/similarity_matrix.json"),
      fetch("/data/index_mapping.json"),
    ])
      .then(([attentionRes, simMatrixRes, indexMapRes]) =>
        Promise.all([attentionRes.json(), simMatrixRes.json(), indexMapRes.json()])
      )
      .then(([attentionData, simMatrixData, indexMapData]) => {
        setAttentionMaps(attentionData.attention_maps);
        setSimilarityMatrix(simMatrixData);
        setIndexMap(indexMapData);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Failed to load data:", error);
        setLoading(false);
      });
  }, []);

  const handleImageChange = (event) => {
    const newImageIndex = event.target.value;
    setSelectedImageIndex(newImageIndex);

    const relatedMaps = attentionMaps.filter(
      (map) => map.image_index === newImageIndex
    );

    setSelectedTextIndex(relatedMaps.length > 0 ? relatedMaps[0].text_index : 0);
  };

  const handleTextChange = (event) => {
    setSelectedTextIndex(event.target.value);
  };

  const uniqueImageIndices = [...new Set(attentionMaps.map((map) => map.image_index))];
  const textOptions = attentionMaps.filter(
    (map) => map.image_index === selectedImageIndex
  );
  const selectedAttentionMap = attentionMaps.find(
    (map) =>
      map.image_index === selectedImageIndex &&
      map.text_index === selectedTextIndex
  );

  // Find the similarity score from the similarity matrix
  const similarityScore = similarityMatrix[selectedImageIndex]?.[selectedTextIndex];

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (attentionMaps.length === 0) {
    return (
      <Typography variant="h6" color="error" mt={4} align="center">
        No attention maps found.
      </Typography>
    );
  }

  return (
    <Box sx={{ p: 4, maxWidth: 1000, mx: "auto" }}>
      <Typography
        variant="h5"
        sx={{
          fontWeight: 600,
          textAlign: "center",
          color: "#333",
        }}
      >
        <IconButton
          sx={{
            marginLeft: 2,
            backgroundColor: "#fffff"
          }}
        >
          <Box sx={{ position: 'relative' }}>
            <SearchIcon sx={{ fontSize: 60, color: "#1976d2" }} />
            <VisibilityIcon
              sx={{
                position: "absolute",
                top: "40%",
                left: "40%",
                transform: "translate(-50%, -50%)",
                fontSize: 20,
                color: "#1976d2",
              }}
            />
          </Box>
        </IconButton>
        Attention mapping viewer
        <Typography
          variant="h5"
          sx={{
            fontSize: "1rem",
            fontWeight: 100,
            textAlign: "center",
            color: "#333",
          }}
        >
          Attention maps are created by computing attention scores that highlight important regions of input.
          These scores are visualized as heatmaps, helping to interpret and understand the model's focus, enhancing model transparency and trust.
          <Box component="span" display="block" marginBottom={2}/>
          Please select the image and text options to see the attention mappings.
          <Box component="span" display="block" marginBottom={2}/>
        </Typography>
      </Typography>

      <Paper
        elevation={3}
        sx={{
          p: 4,
          borderRadius: 4,
          mb: 6,
          backgroundColor: "#fafafa",
        }}
      >
        <Stack direction={{ xs: "column", sm: "row" }} spacing={4}>
          <FormControl fullWidth>
            <InputLabel>Image</InputLabel>
            <Select
              value={selectedImageIndex}
              label="Image"
              onChange={handleImageChange}
            >
              {uniqueImageIndices.map((index) => (
                <MenuItem key={index} value={index}>
                  📷 Image {index}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth>
            <InputLabel>Text</InputLabel>
            <Select
              value={selectedTextIndex}
              label="Text"
              onChange={handleTextChange}
            >
              {textOptions.map((map) => (
                <MenuItem key={map.text_index} value={map.text_index}>
                  📝 {map.text}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>

        {selectedAttentionMap && (
          <Stack
            direction="column"
            alignItems="center"
            justifyContent="center"
            spacing={4}
            margin={15}
          >
            <Card
              sx={{
                transition: "transform 0.35s ease, box-shadow 0.35s ease",
                borderRadius: 4,
                overflow: "hidden",
                "&:hover": {
                  transform: "scale(1.4)",
                  cursor: "zoom-in",
                  boxShadow: "0 8px 24px rgba(0,0,0,0.3)",
                },
              }}
            >{/* Display Similarity Score */}
            <Typography variant="body1" color="textPrimary" align="center" padding={2}>
              Similarity Score: {similarityScore ? similarityScore.toFixed(4) : "N/A"}
            </Typography>
              <CardMedia
                component="img"
                image={`data:image/png;base64,${selectedAttentionMap.attention_map_base64}`}
                alt="Attention Map"
                sx={{
                  width: 500,
                  height: "auto",
                  p: 1,
                  padding: "20px"
                }}
              />
              
            </Card>
          </Stack>
        )}
      </Paper>
    </Box>
  );
}

export default MappingPage;
