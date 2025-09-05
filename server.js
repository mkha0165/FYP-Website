/*
  Purpose: Node.js + Express backend for authentication & static frontend hosting.
  Features:
    - Connects to MongoDB Atlas.
    - Provides `/signup` endpoint for user registration (with bcrypt password hashing).
    - Provides `/login` endpoint for user authentication.
    - Serves static frontend files (index.html, upload.html, CSS, JS, etc.).
    - Ready for cloud deployment (Render, Railway, Heroku, etc.).
*/

const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const cors = require("cors");
const bcrypt = require("bcryptjs");
const path = require("path");

const app = express();

// ===== Middleware =====
app.use(cors());                 // Allows frontend requests from same/different domain
app.use(bodyParser.json());       // Parses JSON request bodies
app.use(express.static(path.join(__dirname))); // Serve static files from project root

// ===== MongoDB Atlas Connection =====
mongoose.connect(
  "mongodb+srv://jkgang432:Jk_gang123@cluster0.lvpe7rz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
  {
    useNewUrlParser: true,
    useUnifiedTopology: true
  }
)
.then(() => console.log("MongoDB Atlas Connected"))
.catch(err => console.error("MongoDB Connection Error:", err));

// ===== Schema & Model =====
const UserSchema = new mongoose.Schema({
  username: { type: String, unique: true, required: true },
  email:    { type: String, unique: true, required: true },
  password: { type: String, required: true }
});

const User = mongoose.model("User", UserSchema);

// ===== Signup Endpoint =====
app.post("/signup", async (req, res) => {
  try {
    const { username, email, password } = req.body;

    // Validate fields
    if (!username || !email || !password) {
      return res.status(400).json({ error: "All fields are required" });
    }

    // Prevent duplicate accounts
    const existingUser = await User.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(400).json({ error: "User already exists" });
    }

    // Hash password before saving
    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({ username, email, password: hashedPassword });
    await user.save();

    res.json({ message: "User registered successfully!" });
  } catch (err) {
    console.error("Signup error:", err.message);
    res.status(500).json({ error: "Server error", details: err.message });
  }
});

// ===== Login Endpoint =====
app.post("/login", async (req, res) => {
  try {
    const { identifier, password } = req.body;
    console.log("Login attempt:", { identifier });

    // Check user by email OR username
    const user = await User.findOne({
      $or: [{ email: identifier }, { username: identifier }]
    });

    if (!user) {
      console.log("User not found");
      return res.status(400).json({ success: false, message: "User not found. Please sign up first." });
    }

    // Compare entered password with hashed password
    const isMatch = await bcrypt.compare(password, user.password);
    console.log("Password match:", isMatch);

    if (!isMatch) {
      return res.status(400).json({ success: false, message: "Invalid password." });
    }

    res.json({ success: true, message: "Login successful!" });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

// ===== Catch-All Route (serves index.html for unknown routes) =====
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// ===== Run Server =====
const PORT = process.env.PORT || 5000; // Render/Heroku assign PORT dynamically
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
