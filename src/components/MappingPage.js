import React, { useState, useEffect } from "react";
import {
  MenuItem,
  FormControl,
  Select,
  InputLabel,
  Box,
  Typography,
  Card,
  CardMedia,
  Alert,
} from "@mui/material";

function MappingPage() {
  const [dropdown1Value, setDropdown1Value] = useState("");
  const [dropdown2Value, setDropdown2Value] = useState("");
  const [error, setError] = useState("");

  const handleDropdown1Change = (event) => {
    setDropdown1Value(event.target.value);
  };

  const handleDropdown2Change = (event) => {
    setDropdown2Value(event.target.value);
  };

  const customData = [
    ["Dog Text", "/dog.jpg", "text"],
    ["Cat Text", "/cat.jpg", "text"],
  ];

  const imageOptions = {
    option1: "/dog.jpg",
    option2: "/cat.jpg",
  };

  const textOptions = {
    optionA: "Dog Text",
    optionB: "Cat Text",
  };

  const isValidSelection = () => {
    const selectedImage = imageOptions[dropdown1Value];
    const selectedText = textOptions[dropdown2Value];

    const matchedItem = customData.find(
      (item) => item[0] === selectedText && item[1] === selectedImage
    );

    if (matchedItem) {
      setError("");
      return true;
    } else {
      setError("Error: The selected image and text do not match.");
      return false;
    }
  };

  useEffect(() => {
    if (dropdown1Value && dropdown2Value) {
      isValidSelection();
    }
  }, [dropdown1Value, dropdown2Value]);

  return (
    <div style={{ position: "relative" }}>
      <Box>
        <FormControl fullWidth sx={{ mt: 4 }}>
          <InputLabel id="dropdown1-label">Select Image</InputLabel>
          <Select
            labelId="dropdown1-label"
            value={dropdown1Value}
            onChange={handleDropdown1Change}
            label="Select Image"
          >
            <MenuItem value="option1">Dog Text</MenuItem>
            <MenuItem value="option2">Cat Text</MenuItem>
          </Select>
        </FormControl>

        <FormControl fullWidth sx={{ mt: 4 }}>
          <InputLabel id="dropdown2-label">Select Text</InputLabel>
          <Select
            labelId="dropdown2-label"
            value={dropdown2Value}
            onChange={handleDropdown2Change}
            label="Select Text"
          >
            <MenuItem value="optionA">Dog Image</MenuItem>
            <MenuItem value="optionB">Cat Image</MenuItem>
          </Select>
        </FormControl>

        {error && <Alert severity="error">{error}</Alert>}

        {dropdown1Value && dropdown2Value && !error && (
          <Box sx={{ mt: 4, textAlign: "center" }}>
            <Typography variant="h5" gutterBottom>
              {textOptions[dropdown2Value]}
            </Typography>

            <Card sx={{ maxWidth: 345, margin: "auto" }}>
              <CardMedia
                component="img"
                height="300"
                image={imageOptions[dropdown1Value]}
                alt="Selected Image"
              />
            </Card>
          </Box>
        )}
      </Box>
    </div>
  );
}

export default MappingPage;
