import { Routes, Route } from 'react-router-dom';

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