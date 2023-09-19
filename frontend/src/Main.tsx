import React, { useEffect, useState } from 'react';
import { Typography } from '@mui/material';
import logo from './robot.svg';
import { io, Socket } from 'socket.io-client';

import Box from '@mui/material/Box';


let socket: Socket

function Main() {
    const [message, setMessage] = useState("")
    const [serverRunning, setServerRunning] = useState(false)
    const [sleeping, setSleeping] = useState(false)

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
            console.log("Message ", message)
            setMessage(message);
        })

        socket.on("sleeping", (sleeping) => {
            if (sleeping === "True") {
                setSleeping(true)
                setMessage("💤")
            } else {
                setSleeping(false)
                setMessage("Hello")
            }

        })

        socket.on("server_status", (status) => {
            console.log("Got status ", status)

            setServerRunning(true)
            setSleeping(status["sleeping"] === "True" ? true : false)
            if (status["sleeping"] === "True") {
                setMessage("💤")
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
            <Box height="500px" marginTop="100px" marginLeft="50px" marginRight="50px" display="flex"
                flexDirection="column"
                justifyContent="center">

                <img src={logo} alt="logo" style={{ height: "100px" }} />
                {message && (
                    <Typography variant={message?.includes("\n") ? "h6" : "h2"} component="div" gutterBottom>
                        {message.split("\n").map((line) => (
                            <p style={{ textAlign: "center" }}>{line} </p>
                        ))}


                    </Typography>
                )}
            </Box>
        </Box >


    )
}

export default Main;