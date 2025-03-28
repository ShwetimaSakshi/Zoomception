import React, { useState } from "react";
import { Container, Typography, AppBar, Toolbar, IconButton, MenuItem, FormControl, Select, InputLabel } from "@mui/material";
import { Link } from "react-router-dom";
import HomeIcon from "@mui/icons-material/Home";

// Styling for dark blue background
const darkBlue = "#003366";

function AttentionPage() {
  const [dropdown1Value, setDropdown1Value] = useState("");
  const [dropdown2Value, setDropdown2Value] = useState("");

  const handleDropdown1Change = (event) => {
    setDropdown1Value(event.target.value);
  };

  const handleDropdown2Change = (event) => {
    setDropdown2Value(event.target.value);
  };

  return (
    <div>
      {/* Menu Bar at the top with a dark blue background */}
      <AppBar position="sticky" sx={{ backgroundColor: darkBlue }}>
        <Toolbar>
          {/* Home Icon on the left */}
          <IconButton edge="start" color="inherit" component={Link} to="/" aria-label="home">
            <HomeIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Content of the AttentionPage */}
      <Container sx={{ mt: 4 }}>
        <Typography variant="h3" gutterBottom>
          Explore Model
        </Typography>
        <Typography variant="body1" paragraph>
          This page is for exploring the model. Here we can include any content related to the model, its features, and visualizations.
        </Typography>

        {/* Select Image */}
        <FormControl fullWidth sx={{ mt: 4 }}>
          <InputLabel id="dropdown1-label">Select Image</InputLabel>
          <Select
            labelId="dropdown1-label"
            value={dropdown1Value}
            onChange={handleDropdown1Change}
            label="Dropdown 1"
          >
            <MenuItem value="option1">Option 1</MenuItem>
            <MenuItem value="option2">Option 2</MenuItem>
            <MenuItem value="option3">Option 3</MenuItem>
          </Select>
        </FormControl>

        {/* Select Text */}
        <FormControl fullWidth sx={{ mt: 4 }}>
          <InputLabel id="dropdown2-label">Select Image</InputLabel>
          <Select
            labelId="dropdown2-label"
            value={dropdown2Value}
            onChange={handleDropdown2Change}
            label="Dropdown 2"
          >
            <MenuItem value="optionA">Option A</MenuItem>
            <MenuItem value="optionB">Option B</MenuItem>
            <MenuItem value="optionC">Option C</MenuItem>
          </Select>
        </FormControl>
      </Container>
    </div>
  );
}

export default AttentionPage;
