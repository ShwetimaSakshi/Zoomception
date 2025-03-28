import React from "react";
import { Container, Typography, AppBar, Toolbar, IconButton } from "@mui/material";
import { Link } from "react-router-dom";
import HomeIcon from "@mui/icons-material/Home";

// Styling for dark blue background
const darkBlue = "#003366"; 

function ModelPage() {
  return (
    <div>
      {/* Menu Bar at the top with a dark blue background */}
      <AppBar position="sticky" sx={{ backgroundColor: darkBlue }}>
        <Toolbar>
          {/* Home Icon on the left */}
          <IconButton edge="start" color="inherit" component={Link} to="/" aria-label="home">
            <HomeIcon />
          </IconButton>
          {/* <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Zoomception
          </Typography> */}
        </Toolbar>
      </AppBar>

      {/* Content of the ModelPage */}
      <Container sx={{ mt: 4 }}>
        <Typography variant="h3" gutterBottom>
          Explore Model
        </Typography>
        <Typography variant="body1" paragraph>
          This page is for exploring the model. Here we can include any content related to the model, its features, and visualizations.
        </Typography>
      </Container>
    </div>
  );
}

export default ModelPage;
