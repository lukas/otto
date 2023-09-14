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
import { stat } from 'fs';

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' ? '#1A2027' : '#fff',
  ...theme.typography.body2,
  padding: theme.spacing(2),
  textAlign: 'center',
  color: theme.palette.text.secondary,
}));



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


let socket: Socket

function System() {
  const [serverRunning, setServerRunning] = useState(false)
  const [sleeping, setSleeping] = useState(false)
  const [factCheck, setFactCheck] = useState(false)
  const [promptPresets, setPromptPresets] = useState([] as Array<{ name: string, prompt: string }>)
  const [availableModels, setAvailableModels] = useState([] as Array<{ model_file: string, prompt_generator: string }>)
  const [message, setMessage] = useState("")
  const [prompt, setPrompt] = useState("")
  const [response, setResponse] = useState("")
  const [promptSetup, setPromptSetup] = useState("")
  const [promptSetupIndex, setPromptSetupIndex] = React.useState(0);
  const [llmModelIndex, setLLMModelIndex] = React.useState(0);
  const [userPrompt, setUserPrompt] = useState("")
  const [oldPrompts, setOldPrompts] = useState({})
  const [rawTranscription, setRawTranscription] = useState("")
  const [rawLLM, setRawLLM] = useState("")
  const [tabValue, setTabValue] = React.useState(0);
  const [llmSettings, setLLMSettings] = useState({ temperature: "0.8", n_predict: "10", forceGrammar: true })
  const [functionCalls, setFunctionCalls] = useState([] as Array<string>)
  const [userFunctionCallStr, setUserFunctionCallStr] = useState("")
  const [runTTSFlag, setRunTTSFlag] = React.useState(true)
  const [runLLMFlag, setRunLLMFlag] = React.useState(true)
  const [runSpeakFlag, setRunSpeakFlag] = React.useState(true)
  const [resetDialogFlag, setResetDialogFlag] = React.useState(true)
  const [llmModel, setLLMModel] = React.useState("llama-2-7b-32k-instruct.ggmlv3.q4_1.bin")

  const handleChange = (event: React.SyntheticEvent, newTabValue: number) => {
    setTabValue(newTabValue);
  };



  useEffect(() => {
    socket = io('ws://' + window.location.hostname + ':5001');

    socket.on("tts", (tts) => {
      setMessage(message => (tts as string));
    })

    socket.on("sleeping", (sleeping) => {
      if (sleeping === "True") {
        setSleeping(true)
      } else {
        setSleeping(false)
      }
    })


    socket.on("prompt", (newPrompt) => {
      console.log("NP ", newPrompt)
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

    socket.on("factcheck", (newTruth) => {
      setFactCheck(factCheck => newTruth === "True" ? true : false)
    })

    socket.on("function_call", (newFunctionCall) => {
      const newFunctionCallStr = newFunctionCall["function_name"] + "(" + newFunctionCall["args"].join(", ") + ")"
      setFunctionCalls(functionCalls => [...functionCalls, newFunctionCallStr])
    })

    socket.on("server_status", (status) => {
      console.log("Got status ", status)
      setServerRunning(true)
      setSleeping(status["sleeping"] === "True" ? true : false)
      setRunSpeakFlag(status["speak_flag"] === "True" ? true : false)
      setResetDialogFlag(status["reset_dialog_flag"] === "True" ? true : false)
      setLLMModel(status["llm_model"] ?? "")
      setAvailableModels(status["available_models"] ?? [])
      setPromptPresets(status["prompt_presets"] ?? [])
      if (status["prompt_presets"].length > 0) {
        setPromptSetup(status["prompt_presets"][0].prompt)
      }


    })

    socket.emit("user_prompt", userPrompt)

    socket.emit("request_status");

    // when component unmounts, disconnect
    return (() => {
      socket.disconnect()
    })
  }, [])

  const serverStatus = serverRunning ? sleeping ? "Sleeping" : "Listening" : "Not Running"
  const functionCallStr = functionCalls.join("\n")
  return (



    <div>
      <AppBar position="static" sx={{ flexDirection: "row" }}   >
        <Typography marginLeft={"12px"}>Otto Bot</Typography>
        <Typography marginLeft={"auto"} marginRight={"12px"}>{serverStatus}</Typography>
      </AppBar>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleChange} aria-label="basic tabs example">
          <Tab label="Dialog" {...a11yProps(0)} />
          <Tab label="Raw Logs" {...a11yProps(1)} />
          <Tab label="Transcription Logs" {...a11yProps(2)} />
          <Tab label="LLM Logs" {...a11yProps(3)} />
          <Tab label="Call Logs" {...a11yProps(4)} />


        </Tabs>
      </Box>




      <CustomTabPanel value={tabValue} index={0}>

        <Paper sx={{ m: "12px", p: "12px" }}>
          <Box>
            <Typography variant="h5" sx={{ textAlign: 'center' }}>Bot Setup</Typography>
            {promptPresets.length > 0 ? (
              <FormControl sx={{ mb: 1, width: 300 }}>
                <InputLabel id="preset-label">Preset Prompt</InputLabel>
                <Select

                  labelId="preset-label"
                  style={{ width: "400px", maxHeight: "48px" }}
                  input={<OutlinedInput label="Preset Prompt" />}
                  label="Preset Prompt"
                  value={String(promptSetupIndex)}
                  onChange={(e: SelectChangeEvent) => {
                    setPromptSetupIndex((promptPresetSelectValue) => Number(e.target.value))
                    setPromptSetup(promptPresets[Number(e.target.value)].prompt);
                    socket.emit("set_prompt_setup", promptPresets[Number(e.target.value)].prompt)
                  }}>
                  {promptPresets.map((preset, i) => (
                    <MenuItem key={preset.name} value={i}>{preset.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            ) : (
              <p>No Presets Found from Server</p>
            )}

          </Box>
          <Box maxHeight="500px" sx={{ flexDirection: 'column' }}>
            <TextField multiline minRows={5} sx={{ flexGrow: 1, maxHeight: '440px', overflow: 'auto' }} style={{ width: "100%" }} defaultValue={promptSetup}
              onChange={(e) => { setPromptSetup(e.target.value); socket.emit("set_prompt_setup", e.target.value) }} />
            <FormGroup row sx={{ mt: "8px", height: "60px" }}>
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
        <Button color="inherit" onClick={() => {
          socket.emit("start_automation",
            {
              "run_tts": runTTSFlag, "run_llm": runLLMFlag, "run_speak": runSpeakFlag,
              "reset_dialog": resetDialogFlag, "llm_model": llmModel,
              "llm_settings": llmSettings
            });
        }}>Start Automation</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_automation"); }}>Stop Automation</Button>

        <FormGroup>
          {availableModels.length > 0 ? (
            <FormControl sx={{ mb: 1, width: 300 }}>
              <InputLabel id="preset-label">Model</InputLabel>
              <Select

                labelId="modell"
                style={{ width: "400px", maxHeight: "48px" }}
                input={<OutlinedInput label="Model" />}
                label="Model"
                value={String(llmModelIndex)}
                onChange={(e: SelectChangeEvent) => {
                  setLLMModelIndex(Number(e.target.value))
                  setLLMModel(availableModels[Number(e.target.value)].model_file);
                }}>
                {availableModels.map((model, i) => (
                  <MenuItem key={i} value={i}>{model.model_file}</MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : (
            <p>No Models found from Server</p>
          )}
          < FormControlLabel control={<Switch checked={runTTSFlag} onChange={(v) => setRunTTSFlag(v.target.checked)} />} label="Run Text to Speech" />
          <FormControlLabel control={<Switch checked={runLLMFlag} onChange={(v) => setRunLLMFlag(v.target.checked)} />} label="Run LLM" />
          <FormControlLabel control={<Switch checked={runSpeakFlag} onChange={(v) => setRunSpeakFlag(v.target.checked)} />} label="Run Speaking" />
          <FormControlLabel control={<Switch checked={resetDialogFlag} onChange={(v) => setResetDialogFlag(v.target.checked)} />} label="Reset Dialog Each Time" />
          <FormControlLabel control={<Switch checked={llmSettings.forceGrammar} onChange={(v) => setLLMSettings((llmSettings) => { return { ...llmSettings, forceGrammar: v.target.checked } })} />} label="Force Grammar" />

          <TextField id="outlined-basic" label="Temperature" value={llmSettings.temperature}
            style={{ width: '200px' }}
            onChange={(e) => { }} />
          <TextField style={{ marginTop: "8px", width: '200px' }} id="N Predict" label=" N Predict" value={llmSettings.n_predict}
            onChange={(e) => { setLLMSettings((llmSettings) => { return { ...llmSettings, n_predict: e.target.value } }) }} />
        </FormGroup>
        {
          (prompt !== "") && (
            <Paper sx={{ m: "12px", p: "12px" }} >
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
              <pre>
                {response}
              </pre>

            </Paper>
          )
        }
      </CustomTabPanel>
      <CustomTabPanel value={tabValue} index={2}>
        <Button color="inherit" onClick={() => { socket.emit("start_transcribe"); }}>Start Transcription</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_transcribe"); setRawTranscription(rawTranscription => ""); }}>Stop Transcription</Button>
        {(rawTranscription !== "") && (
          <Paper sx={{ m: "12px" }} style={{ height: "100%", overflow: "auto" }}>
            <Typography variant="h5" sx={{ textAlign: 'center' }}>Raw Response</Typography>
            <pre style={{ height: "500px", overflow: "auto", display: "flex", flexDirection: "column-reverse" }}>
              {rawTranscription}

            </pre>
          </Paper>
        )}
      </CustomTabPanel>

      <CustomTabPanel value={tabValue} index={3} >
        <Button color="inherit" onClick={() => { socket.emit("start_llm"); }}>Start LLM</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_llm"); setRawLLM(rawLLM => ""); }}>Stop LLM</Button>

        {(rawLLM !== "") && (
          <Paper sx={{ m: "12px" }} >
            <Typography variant="h5" sx={{ textAlign: 'center' }}>Raw Response</Typography>
            <pre style={{ height: "500px", overflow: "auto", display: "flex", flexDirection: "column-reverse" }}>
              {rawLLM}

            </pre>
          </Paper>
        )}
      </CustomTabPanel>

      <CustomTabPanel value={tabValue} index={4} >


        <Paper sx={{ m: "12px" }} >
          <Typography variant="h5" sx={{ textAlign: 'center' }}>Call Log</Typography>
          <pre style={{ height: "500px", overflow: "auto", display: "flex", flexDirection: "column-reverse" }}>
            {functionCallStr}

          </pre>
          <FormGroup row sx={{ mt: "8px", height: "60px" }}>
            <TextField style={{ flex: 1 }} id="outlined-basic" label="Manual Call" variant="outlined" value={userFunctionCallStr}
              onChange={(e) => { setUserFunctionCallStr(e.target.value) }} />

            <Button color="inherit" onClick={() => { socket.emit("call", userFunctionCallStr); }}>Call</Button>

          </FormGroup>
        </Paper>

      </CustomTabPanel>

    </div >


  )
}

export default System;
