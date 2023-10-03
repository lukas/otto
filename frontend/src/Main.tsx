import { useEffect, useState } from 'react';
import { Typography } from '@mui/material';
import logo from './robot.svg';
import { io, Socket } from 'socket.io-client';

import Box from '@mui/material/Box';
import VolumeOffIcon from '@mui/icons-material/VolumeOff';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import SettingsIcon from '@mui/icons-material/Settings';
import IconButton from '@mui/material/IconButton';
import { useNavigate } from 'react-router-dom';


let socket: Socket

function Main() {
    const [message, setMessage] = useState("Not Connected to Server")
    const [serverRunning, setServerRunning] = useState(false)
    const [sleeping, setSleeping] = useState(false)
    const [soundFlag, setSoundFlag] = useState(false)
    const [timer, setTimer] = useState(0)
    const navigate = useNavigate();
    const speakFlag = true;
    let utterance: SpeechSynthesisUtterance;
    function newMessage(message: string) {
        setMessage(message);
        if (speakFlag) {
            console.log("Speaking ", message)
            socket.emit("start_speaking");
            utterance = new SpeechSynthesisUtterance(message);
            utterance.onend = () => {
                console.log("Done speaking")
                socket.emit("stop_speaking");
            }
            speechSynthesis.speak(utterance);

        }
    }

    function stopSpeaking() {
        speechSynthesis.cancel();
    }


    useEffect(() => {
        socket = io('ws://' + window.location.hostname + ':5001');
        socket.on("factcheck", (status) => {
            if (status === "start") {
                setMessage("Fact Checking");
            } else if (status === "false") {
                setMessage("False Claim Detected")
            }
        })

        socket.on("transcribe", (transcription) => {
            // setMessage(transcription);
        })

        socket.on("message", (message) => {
            newMessage(message);
        })

        socket.on("timer", (newTime) => {
            setTimer(newTime)
        })

        socket.on("sleeping", (sleeping) => {
            if (sleeping === "True") {
                setSleeping(true)
                setMessage("💤")
            } else {
                setSleeping(false)
                newMessage("Hello")
            }

        })

        socket.on("server_status", (status) => {

            setServerRunning(true)
            setSleeping(status["sleeping"] === "True" ? true : false)
            if (status["sleeping"] === "True") {
                setMessage("💤")
            } else {
                setMessage("...")
            }
        })

        socket.emit("request_status");

        // when component unmounts, disconnect
        return (() => {
            socket.disconnect()
        })
    }, [])

    return (
        <Box alignItems="center"
            justifyContent="center"
            display='flex'
            flexDirection="column"
            height="100%"
        >
            <Box display="flex" width="100%" flexDirection="row-reverse">

                <IconButton aria-label="settings" onClick={() => {
                    navigate('/system');
                }}>
                    <SettingsIcon />
                </IconButton>

                <IconButton aria-label="sound" onClick={() => {
                    setSoundFlag(!soundFlag); stopSpeaking()
                }}>

                    {soundFlag ? <VolumeUpIcon /> : <VolumeOffIcon />}
                </IconButton>
                {(timer != 0) &&
                    <Typography sx={{ pt: '8px', pb: '8px' }} variant="h6" textAlign={{}}  >
                        {new Date(timer * 1000).toISOString().substring(11, 19)}
                    </Typography>
                }


            </Box>

            <Box height="500px" marginTop="100px" marginLeft="50px" marginRight="50px" display="flex"
                flexDirection="column"
                justifyContent="center">

                <img src={logo} alt="logo" style={{ height: "100px" }} />
                {message && (
                    <Typography variant={message?.includes("\n") ? "h6" : "h2"} component="div" gutterBottom>
                        {message.split("\n").map((line, i) => (
                            <p key={"Message line " + i} style={{ textAlign: "center" }}>{line} </p>
                        ))}


                    </Typography>
                )}
            </Box>
        </Box >


    )
}

export default Main;