import React, { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Button from "@mui/material/Button";
import Grid from "@mui/material/Grid";
import { styled } from '@mui/material/styles';
import AppBar from '@mui/material/AppBar';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import FormGroup from '@mui/material/FormGroup';
import { TextareaAutosize } from '@mui/base/TextareaAutosize';
import { io, Socket } from 'socket.io-client';
import MenuItem from '@mui/material/MenuItem';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import InputLabel from '@mui/material/InputLabel';
import OutlinedInput from '@mui/material/OutlinedInput';
import FormControl from '@mui/material/FormControl';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' ? '#1A2027' : '#fff',
  ...theme.typography.body2,
  padding: theme.spacing(2),
  textAlign: 'center',
  color: theme.palette.text.secondary,
}));

const presets = [
  ["Fact Checker",
    `You are a fact checking bot. If the human says a fact and the fact is true,
  you say, true. If the human say a fact that is false, you say false. If you
  are not sure, you say maybe.`],
  ["Octopus",
    `You are Oren the Octopus a friendly chatbot. You are having a conversation 
    with a human child.`],
  ["Evil",
    `You are an evil chatbot that always lies. 
    You are having a conversation with a human but you never tell the truth.`],

]

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function CustomTabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}


// outside of your component, initialize the socket variable
let socket: Socket

function App() {

  const [message, setMessage] = useState("")
  const [prompt, setPrompt] = useState("")
  const [response, setResponse] = useState("")
  const [promptSetup, setPromptSetup] = useState("")
  const [userPrompt, setUserPrompt] = useState("")
  const [oldPrompts, setOldPrompts] = useState({})
  const [rawTranscription, setRawTranscription] = useState("")
  const [rawLLM, setRawLLM] = useState("")
  const [tabValue, setTabValue] = React.useState(0);
  const [promptPresetSelectValue, setPromptPresetSelectValue] = React.useState(0);

  const handleChange = (event: React.SyntheticEvent, newTabValue: number) => {
    setTabValue(newTabValue);
  };



  useEffect(() => {
    socket = io('ws://localhost:5001');

    socket.on("tts", (tts) => {
      setMessage(message => (tts as string));
    })

    socket.on("prompt", (newPrompt) => {
      setPrompt(prompt => (newPrompt as string));
      setResponse(response => "");
    })

    socket.on("response", (token) => {
      setResponse(response => response += token);
    })

    socket.on("prompt_setup", (newPromptSetup) => {
      setPromptSetup(promptSetup => (newPromptSetup as string));
    })

    socket.on("user_prompt", (newUserPrompt) => {
      setUserPrompt(userPrompt => (newUserPrompt as string));
    })

    socket.on("old_prompts", (newOldPrompts) => {
      setOldPrompts(oldPrompts => newOldPrompts)
    })
    socket.on("transcribe_stdout", (newTranscription) => {
      setRawTranscription(rawTranscription => rawTranscription + (newTranscription as string) + "\n");
    })

    socket.on("llm_stdout", (newLLM) => {
      setRawLLM(rawLLM => rawLLM + (newLLM as string) + "\n")
    })

    socket.emit("user_prompt", userPrompt)

    socket.emit("refresh");

    // when component unmounts, disconnect
    return (() => {
      socket.disconnect()
    })
  }, [])





  return (
    <div>
      <AppBar position="static" sx={{ flexDirection: "row" }}>

        <Box>
          <Button color="inherit" onClick={() => { socket.emit("start_auto"); }}>Start</Button>
          <Button color="inherit" onClick={() => { socket.emit("stop_auto"); }}>Stop</Button>
        </Box>

      </AppBar>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleChange} aria-label="basic tabs example">
          <Tab label="Dialog" {...a11yProps(0)} />
          <Tab label="Raw Logs" {...a11yProps(1)} />
          <Tab label="Transcription Logs" {...a11yProps(2)} />
          <Tab label="LLM Logs" {...a11yProps(3)} />

          <Tab label="Settings" {...a11yProps(4)} />

        </Tabs>
      </Box>




      <CustomTabPanel value={tabValue} index={0}>

        <Paper sx={{ m: "12px", p: "12px" }}>
          <Box>
            <Typography variant="h5" sx={{ textAlign: 'center' }}>Bot Setup</Typography>
            <FormControl sx={{ mb: 1, width: 300 }}>
              <InputLabel id="preset-label">Preset Prompt</InputLabel>
              <Select

                labelId="preset-label"
                style={{ width: "400px", maxHeight: "48px" }}
                input={<OutlinedInput label="Preset Prompt" />}
                label="Preset Prompt"
                value={String(promptPresetSelectValue)}
                onChange={(e: SelectChangeEvent) => {
                  setPromptPresetSelectValue((promptPresetSelectValue) => Number(e.target.value))
                  console.log("Changing ", e.target.value)
                  setPromptSetup(presets[Number(e.target.value)][1]);
                  socket.emit("set_prompt_setup", presets[Number(e.target.value)][1])
                }}>
                {presets.map((preset, i) => (


                  <MenuItem key={preset[0]} value={i}>{preset[0]}</MenuItem>
                ))}
              </Select>
            </FormControl>


          </Box>
          <Box>
            <TextField multiline minRows={5} style={{ width: "100%" }} defaultValue={promptSetup}
              onChange={(e) => { setPromptSetup(e.target.value); socket.emit("set_prompt_setup", e.target.value) }} />
            <FormGroup row sx={{ mt: "8px" }}>
              <TextField style={{ flex: 1 }} id="outlined-basic" label="Human Response" variant="outlined" value={userPrompt}
                onChange={(e) => { setUserPrompt(e.target.value) }} />
              <Button color="inherit" onClick={() => { socket.emit("manual_prompt", userPrompt); }}>Chat</Button>
              <Button color="inherit" onClick={() => { socket.emit("stop_talking", userPrompt); }}>Stop</Button>

              <Button color="inherit" onClick={() => { socket.emit("reset_dialog"); }}>Reset</Button>

            </FormGroup>
          </Box>
        </Paper >
        {(userPrompt !== "") && (
          <Paper sx={{ m: "12px", p: "12px" }}>
            <Typography variant="h5" sx={{ textAlign: 'center' }}>Chat</Typography>
            {(("old_prompts" in oldPrompts) && ("old_responses" in oldPrompts)) && (
              ((oldPrompts["old_prompts"] as Array<string>).map((old_prompt: string, i: number) => (
                <p key={i}><b>Human</b> {old_prompt}<br /><b>Bot</b> {(oldPrompts["old_responses"] as Array<string>)[i]}</p>
              ))
              ))
            }
            <p><b>Human</b> {userPrompt}</p>
            {response !== "" && (<p><b>Bot</b> {response}</p>)}
          </Paper>
        )}
        <Paper sx={{ m: "12px", p: "12px" }}>
          <Typography variant="h5" sx={{ textAlign: 'center' }}>Listening Transcript</Typography>
          <pre>
            {message}
          </pre>
        </Paper>

      </CustomTabPanel>

      <CustomTabPanel value={tabValue} index={1}>
        <Button color="inherit" onClick={() => { socket.emit("start_automation"); }}>Start Automation</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_automation"); }}>Stop Automation</Button>

        {
          (prompt !== "") && (
            <Paper sx={{ m: "12px", p: "12px" }}>
              <Typography variant="h5" sx={{ textAlign: 'center' }}>Raw Prompt</Typography>
              <pre>
                {prompt}
              </pre>


            </Paper>
          )
        }
        {
          (response !== "") && (
            <Paper sx={{ m: "12px" }}>
              <Typography variant="h5" sx={{ textAlign: 'center' }}>Raw Response</Typography>

              {response}


            </Paper>
          )
        }
      </CustomTabPanel>
      <CustomTabPanel value={tabValue} index={2}>
        <Button color="inherit" onClick={() => { socket.emit("start_transcribe"); }}>Start Transcription</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_transcribe"); setRawTranscription(rawTranscription => ""); }}>Stop Transcription</Button>
        {(rawTranscription !== "") && (
          <Paper sx={{ m: "12px" }}>
            <Typography variant="h5" sx={{ textAlign: 'center' }}>Raw Response</Typography>
            <pre>
              {rawTranscription}

            </pre>
          </Paper>
        )}
      </CustomTabPanel>

      <CustomTabPanel value={tabValue} index={3}>
        <Button color="inherit" onClick={() => { socket.emit("start_llm"); }}>Start LLM</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_llm"); setRawLLM(rawLLM => ""); }}>Stop LLM</Button>
        {(rawLLM !== "") && (
          <Paper sx={{ m: "12px" }}>
            <Typography variant="h5" sx={{ textAlign: 'center' }}>Raw Response</Typography>
            <pre>
              {rawLLM}

            </pre>
          </Paper>
        )}
      </CustomTabPanel>

      <CustomTabPanel value={tabValue} index={4}>
        <FormGroup>
          <FormControlLabel control={<Switch defaultChecked />} label="Run Text to Speech" />
          <FormControlLabel control={<Switch defaultChecked />} label="Run LLM" />
          <FormControlLabel control={<Switch defaultChecked />} label="Run Speaking" />
        </FormGroup>
        <Button color="inherit" onClick={() => { socket.emit("reset_dialog"); }}>Reset Dialog</Button>

      </CustomTabPanel>


    </div >
  );
}

export default App;
