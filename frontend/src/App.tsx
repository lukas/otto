import * as React from 'react';
import { Routes, Route } from 'react-router-dom';
import { io, Socket } from 'socket.io-client';


import Main from './Main';
import System from './System';


function App() {
    return (
        <div className="App">
            <Routes>
                <Route path="/" element={<Main />} />
                <Route path="system" element={<System />} />

            </Routes>
        </div>
    );
}

export default App