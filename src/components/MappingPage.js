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
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/data/attention_results.json")
      .then((response) => response.json())
      .then((data) => {
        setAttentionMaps(data.attention_maps);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Failed to load attention maps:", error);
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
    <Box sx={{ p: 4, maxWidth: 1000, mx: "auto"}}>
      <Typography
        variant="h5"
        sx={{
          fontWeight: 600,
          textAlign: "center",
          // mb: 4,
          color: "#333",
        }}
      >
        <IconButton
          sx={{
            marginLeft: 2,
            backgroundColor: "#fffff"
          }} Attention Map Viewer
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
        <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
          Select Options
        </Typography>

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
          >
            <CardMedia
              component="img"
              image={`data:image/png;base64,${selectedAttentionMap.attention_map_base64}`}
              alt="Attention Map"
              sx={{
                width: 500,
                height: "auto",
                p: 1,
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
