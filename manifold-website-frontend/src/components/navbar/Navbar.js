// manifold-website-frontend/src/components/navbar/Navbar.js
"use client";

import React, { useState } from "react";
import UploadModal from "../upload/UploadModal";

export default function Navbar() {
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  return (
    <>
      <nav className="bg-gray-900 text-white shadow-lg">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold">Manifold</h1>
          </div>
          
          <ul className="flex items-center space-x-6">
            <li>
              <button
                onClick={() => setIsUploadOpen(true)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg 
                         transition-colors duration-200 font-medium"
              >
                Upload
              </button>
            </li>
          </ul>
        </div>
      </nav>

      {/* Upload Modal */}
      <UploadModal 
        isOpen={isUploadOpen} 
        onClose={() => setIsUploadOpen(false)} 
      />
    </>
  );
}