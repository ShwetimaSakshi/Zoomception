import React, { useState } from "react";
import {
  Container,
  AppBar,
  Toolbar,
  IconButton,
  Button,
  Stack,
} from "@mui/material";
import { Link } from "react-router-dom";
import HomeIcon from "@mui/icons-material/Home";
import EmbeddingsPage from "./EmbeddingsPage"; 
import MappingPage from "./MappingPage"; 

// Styling
const darkBlue = "#003366";

function AttentionPage() {
  const [activeSection, setActiveSection] = useState("attention");

  return (
    <div>
      <AppBar position="sticky" sx={{ backgroundColor: darkBlue }}>
        <Toolbar>
          <IconButton
          edge="start" 
          color="inherit" 
          component={Link} 
          to="/" 
          aria-label="home"
          sx={{ 
            mr: 2,
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.1)'
            }
          }}
          >
            <HomeIcon />
          </IconButton>

          <Stack direction="row" spacing={2} sx={{ ml: 2 }}>
            <Button
              variant={activeSection === "attention" ? "contained" : "outlined"}
              color="secondary"
              sx={{color:"white"}}
              onClick={() => setActiveSection("attention")}
            >
              Attention Mapping
            </Button>
            <Button
              variant={activeSection === "embeddings" ? "contained" : "outlined"}
              color="secondary"
              sx={{color:"white"}}
              onClick={() => setActiveSection("embeddings")}
            >
              Embeddings
            </Button>
          </Stack>
        </Toolbar>
      </AppBar>

     
      <Container sx={{ mt: 4 }}>
        {activeSection === "attention" && <MappingPage/>}

        {activeSection === "embeddings" && <EmbeddingsPage />}

      </Container>
    </div>
  );
}

export default AttentionPage;
