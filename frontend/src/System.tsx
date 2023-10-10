import React, { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Button from "@mui/material/Button";
import AppBar from '@mui/material/AppBar';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import FormGroup from '@mui/material/FormGroup';
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
import Accordion from '@mui/joy/Accordion';

import AccordionDetails from '@mui/joy/AccordionDetails';
import AccordionGroup from '@mui/joy/AccordionGroup';
import AccordionSummary from '@mui/joy/AccordionSummary';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';

// const Item = styled(Paper)(({ theme }) => ({
//   backgroundColor: theme.palette.mode === 'dark' ? '#1A2027' : '#fff',
//   ...theme.typography.body2,
//   padding: theme.spacing(2),
//   textAlign: 'center',
//   color: theme.palette.text.secondary,
// }));



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
  const [promptPresets, setPromptPresets] = useState([] as Array<{ name: string, prompt: string }>)
  const [skills, setSkills] = useState({} as { [Name: string]: string })

  const [availableLLMModels, setAvailableLLMModels] = useState([] as Array<{ model: string, prompt_generator: string }>)
  const [availableTranscribeModels, setAvailableTranscribeModels] = useState([] as Array<{ model: string }>)
  const [llmModelIndex, setLLMModelIndex] = React.useState(0);
  const [transcribeModelIndex, setTranscribeModelIndex] = React.useState(0);

  const [overview, setOvervierw] = useState("")
  const [prompt, setPrompt] = useState("")
  const [response, setResponse] = useState("")
  const [promptSetup, setPromptSetup] = useState("")
  const [promptSetupIndex, setPromptSetupIndex] = React.useState(0);
  const [userPrompt, setUserPrompt] = useState("")
  const [oldPrompts, setOldPrompts] = useState({})
  const [rawTranscription, setRawTranscription] = useState("")
  const [transcriptionLog, setTranscriptionLog] = useState("")
  const [rawLLM, setRawLLM] = useState("")
  const [tabValue, setTabValue] = React.useState(0);
  const [llmSettings, setLLMSettings] = useState({ model: "", temperature: "0.8", n_predict: "10", force_grammar: true })
  const [transcribeSettings, setTranscribeSettings] = useState({ model: "", threads: "4", step: "3000", length: "10000", keep: "200", "max-tokens": "32", "vad-thold": "0.6", "freq-thold": "100.0", "speed-up": false, "no-fallback": false })

  const [skillLog, setSkillLog] = useState("")
  const [userFunctionCallStr, setUserFunctionCallStr] = useState("")
  const [runTranscribeFlag, setRunTranscribeFlag] = React.useState(true)
  const [runLLMFlag, setRunLLMFlag] = React.useState(true)
  const [runSpeakFlag, setRunSpeakFlag] = React.useState(true)
  const [resetDialogFlag, setResetDialogFlag] = React.useState(true)



  const handleChange = (event: React.SyntheticEvent, newTabValue: number) => {
    setTabValue(newTabValue);
  };


  function overviewPanel() {
    return (
      <CustomTabPanel value={tabValue} index={0}>
        <AccordionGroup>
          <Accordion>
            <AccordionSummary>Settings</AccordionSummary>
            <AccordionDetails>
              {overviewSettingsPanel()}
            </AccordionDetails>
          </Accordion>
          <Accordion>
            <AccordionSummary>Logs</AccordionSummary>
            <AccordionDetails>
              {overviewLogsPanel()}
            </AccordionDetails>
          </Accordion>
        </AccordionGroup>




      </CustomTabPanel>
    )
  }

  function llmPanel() {
    return (
      <CustomTabPanel value={tabValue} index={2} >
        <AccordionGroup>
          <Accordion>
            <AccordionSummary>Settings</AccordionSummary>
            <AccordionDetails>
              {llmSettingsPanel()}
            </AccordionDetails>
          </Accordion>
          <Accordion>
            <AccordionSummary>Call LLM Manually</AccordionSummary>
            <AccordionDetails>
              {callLlmManually()}
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary>Logs</AccordionSummary>
            <AccordionDetails>
              {llmLogsPanel()}
            </AccordionDetails>
          </Accordion>
          <Accordion>
            <AccordionSummary>System Logs</AccordionSummary>
            <AccordionDetails>
              {llmSystemLogsPanel()}
            </AccordionDetails>
          </Accordion>
        </AccordionGroup>
      </CustomTabPanel>
    )
  }

  function callLlmManually() {
    return (
      <Paper sx={{ m: "12px", p: "12px" }}>
        <Box>
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
        <Box maxHeight="400px" sx={{ flexDirection: 'column' }}>
          <TextField multiline minRows={5} sx={{ flexGrow: 1, maxHeight: '340px', overflow: 'auto' }} style={{ width: "100%" }} defaultValue={promptSetup}
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
    )
  }

  function llmLogsPanel() {
    return (
      <Box>
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
      </Box>
    )
  }


  function llmSystemLogsPanel() {
    return (
      <Box>

        {
          (rawLLM !== "") && (
            <Paper sx={{ m: "12px" }} >
              <Typography variant="h5" sx={{ textAlign: 'center' }}>LLM Logs</Typography>
              <pre style={{ height: "500px", overflow: "auto", display: "flex", flexDirection: "column-reverse" }}>
                {rawLLM}

              </pre>
            </Paper>
          )
        }
      </Box>)
  }

  function llmSettingsPanel() {
    return (
      <Box>
        <Button color="inherit" onClick={() => { socket.emit("start_llm", llmSettings); }}>Start LLM</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_llm"); setRawLLM(rawLLM => ""); }}>Stop LLM</Button>
        <FormGroup row sx={{ mt: "8px", display: 'flex', flexDirection: 'column' }}>
          {availableLLMModels.length > 0 ? (
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
                  setLLMSettings((llmSettings) => { return { ...llmSettings, model: availableLLMModels[Number(e.target.value)].model } })
                }}>
                {availableLLMModels.map((model, i) => (
                  <MenuItem key={i} value={i}>{model.model}</MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : (
            <p>No Models found from Server</p>
          )}

          <FormControlLabel control={<Switch checked={llmSettings.force_grammar} onChange={(v) => setLLMSettings((llmSettings) => { return { ...llmSettings, force_grammar: v.target.checked } })} />} label="Force Grammar" />
          <TextField id="outlined-basic" label="Temperature" value={llmSettings.temperature}
            style={{ width: '200px' }}
            onChange={(e) => { }} />
          <TextField style={{ marginTop: "8px", width: '200px' }} id="N Predict" label=" N Predict" value={llmSettings.n_predict}
            onChange={(e) => { setLLMSettings((llmSettings) => { return { ...llmSettings, n_predict: e.target.value } }) }} />
        </FormGroup >
      </Box >
    )
  }


  function overviewLogsPanel() {
    return (
      <Box>

        <Paper sx={{ m: "12px" }} >
          <pre style={{ height: "500px", overflow: "auto", display: "flex", flexDirection: "column-reverse" }}>
            {overview}

          </pre>
        </Paper>


      </Box>
    )
  }


  function overviewSettingsPanel() {
    return (
      <Box>
        <Button color="inherit" onClick={() => {
          socket.emit("start_automation",
            {
              "run_transcribe": runTranscribeFlag, "run_llm": runLLMFlag, "run_speak": runSpeakFlag,
              "reset_dialog": resetDialogFlag,
              "llm_settings": llmSettings,
              "transcribe_settings": transcribeSettings
            });
        }}>Start Automation</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_automation"); }}>Stop Automation</Button>

        <FormGroup>

          <FormControlLabel control={<Switch checked={runTranscribeFlag} onChange={(v) => setRunTranscribeFlag(v.target.checked)} />} label="Run Transcription" />
          <FormControlLabel control={<Switch checked={runLLMFlag} onChange={(v) => setRunLLMFlag(v.target.checked)} />} label="Run LLM" />
          <FormControlLabel control={<Switch checked={runSpeakFlag} onChange={(v) => setRunSpeakFlag(v.target.checked)} />} label="Run Speaking" />
          <FormControlLabel control={<Switch checked={resetDialogFlag} onChange={(v) => setResetDialogFlag(v.target.checked)} />} label="Reset Dialog Each Time" />

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
      </Box>
    )
  }

  function transcriptionPanel() {
    return (
      <CustomTabPanel value={tabValue} index={1}>
        <AccordionGroup>
          <Accordion>
            <AccordionSummary>Settings</AccordionSummary>
            <AccordionDetails>
              {transcriptionSettingsPanel()}
            </AccordionDetails>
          </Accordion>
          <Accordion>
            <AccordionSummary>Logs</AccordionSummary>
            <AccordionDetails>
              {transcriptionLogsPanel()}
            </AccordionDetails>
          </Accordion>
          <Accordion>
            <AccordionSummary>System Logs</AccordionSummary>
            <AccordionDetails>
              {transcriptionSystemLogsPanel()}
            </AccordionDetails>
          </Accordion>
        </AccordionGroup >
      </CustomTabPanel>


    )
  }

  function transcriptionLogsPanel() {
    return (
      <Paper sx={{ m: "12px", p: "12px", maxHeight: '400px', flexDirection: 'column' }} >
        <Typography variant="h5" sx={{ textAlign: 'center' }}>Listening Transcript</Typography>
        <pre style={{ overflow: 'auto', height: '300px', display: "flex", flexDirection: "column-reverse" }}>
          {transcriptionLog}
        </pre>
      </Paper>
    )
  }

  function transcriptionSystemLogsPanel() {
    return (
      <Box>
        {
          (rawTranscription !== "") && (
            <Paper sx={{ m: "12px" }} style={{ height: "100%", overflow: "auto" }}>
              <Typography variant="h5" sx={{ textAlign: 'center' }}>Transcription Logs</Typography>
              <pre style={{ height: "500px", overflow: "auto", display: "flex", flexDirection: "column-reverse" }}>
                {rawTranscription}

              </pre>
            </Paper>
          )
        }
      </Box>
    )
  }

  function transcriptionSettingsPanel() {
    return (
      <Box>
        <Button color="inherit" onClick={() => { socket.emit("start_transcribe", transcribeSettings); }}>Start Transcription</Button>
        <Button color="inherit" onClick={() => { socket.emit("stop_transcribe"); setRawTranscription(rawTranscription => ""); }}>Stop Transcription</Button>
        <FormGroup row sx={{ mt: "8px", display: 'flex', flexDirection: 'column' }}>
          {availableTranscribeModels.length > 0 ? (
            <FormControl sx={{ mb: 1, width: 300 }}>
              <InputLabel id="preset-label">Model</InputLabel>
              <Select

                labelId="modell"
                style={{ width: "400px", maxHeight: "48px" }}
                input={<OutlinedInput label="Model" />}
                label="Model"
                value={String(transcribeModelIndex)}
                onChange={(e: SelectChangeEvent) => {
                  setTranscribeModelIndex(Number(e.target.value))
                  setTranscribeSettings((t) => {
                    return { ...t, model: availableTranscribeModels[Number(e.target.value)].model }
                  })
                }}>
                {availableTranscribeModels.map((model, i) => (
                  <MenuItem key={i} value={i}>{model.model}</MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : (
            <p>No Transcribe Models found from Server</p>
          )
          }

          <TextField style={{ marginTop: "8px", width: '200px' }} id="Number Threads" label="Number Threads" value={transcribeSettings.threads}
            onChange={(e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, threads: e.target.value } }) }} />
          <TextField style={{ marginTop: "8px", width: '200px' }} id="Step" label="Step (ms)" value={transcribeSettings.step}
            onChange={(e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, step: e.target.value } }) }} />
          <TextField style={{ marginTop: "8px", width: '200px' }} id="Length" label="Length (ms)" value={transcribeSettings.length}
            onChange={(e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, length: e.target.value } }) }} />
          <TextField style={{ marginTop: "8px", width: '200px' }} id="Keep" label="Keep (ms)" value={transcribeSettings.keep}
            onChange={(e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, keep: e.target.value } }) }} />
          <TextField style={{ marginTop: "8px", width: '200px' }} id="Max Tokens" label="Max Tokens" value={transcribeSettings['max-tokens']}
            onChange={(e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, 'max-tokens': e.target.value } }) }} />
          <TextField style={{ marginTop: "8px", width: '200px' }} id="Vad" label="Vad Threshold" value={transcribeSettings['vad-thold']}
            onChange={(e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, 'val-thold': e.target.value } }) }} />
          <TextField style={{ marginTop: "8px", width: '200px' }} id="Freq" label="Freq Threshold" value={transcribeSettings.keep}
            onChange={(e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, 'freq-thold': e.target.value } }) }} />
          <FormControlLabel control={<Switch checked={transcribeSettings['no-fallback']} onChange={
            (e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, 'no-fallback': e.target.checked } }) }} />} label="No Fallback" />
          <FormControlLabel control={<Switch checked={transcribeSettings['speed-up']} onChange={
            (e) => { setTranscribeSettings((transcribeSettings) => { return { ...transcribeSettings, 'speed-up': e.target.checked } }) }} />} label="2x Speed" />

        </FormGroup >
      </Box>

    )

  }

  function skillPanel() {
    return (
      <CustomTabPanel value={tabValue} index={3}>
        <AccordionGroup>
          <Accordion>
            <AccordionSummary>Settings</AccordionSummary>
            <AccordionDetails>
              {skillSettingsPanel()}
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary>Logs</AccordionSummary>
            <AccordionDetails>
              {skillLogsPanel()}
            </AccordionDetails>
          </Accordion>

        </AccordionGroup >
      </CustomTabPanel>
    )
  }

  function skillLogsPanel() {
    return (
      <Paper sx={{ m: "12px", p: "12px", maxHeight: '400px', flexDirection: 'column' }} >
        <pre style={{ overflow: 'auto', height: '300px', display: "flex", flexDirection: "column-reverse" }}>
          {skillLog}
        </pre>
        <FormGroup row sx={{ mt: "8px", height: "60px" }}>
          <TextField style={{ flex: 1 }} id="outlined-basic" label="Manual Call" variant="outlined" value={userFunctionCallStr}
            onChange={(e) => { setUserFunctionCallStr(e.target.value) }} />

          <Button color="inherit" onClick={() => { socket.emit("call", userFunctionCallStr); }}>Call</Button>

        </FormGroup>
      </Paper>
    )
  }

  function skillSettingsPanel() {
    return (
      <Box>
        <Typography variant="h5" sx={{ textAlign: 'center' }}>Available Skills</Typography>
        <List>
          {
            Object.keys(skills).map((skillName, i) => (
              <ListItem key={`skill-list-${i}`}>{skillName}: {skills[skillName]}</ListItem>
            ))
          }
        </List>
      </Box>
    )
  }

  useEffect(() => {
    socket = io('ws://' + window.location.hostname + ':5001');
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
    socket.on("raw_transcription", (newTranscription) => {
      setRawTranscription(rawTranscription => rawTranscription + (newTranscription as string) + "\n");
    })

    socket.on("transcribe", (newTranscription) => {
      setTranscriptionLog(transcriptionLog => transcriptionLog + (newTranscription as string) + "\n");
    })
    socket.on("llm_stdout", (newLLM) => {
      setRawLLM(rawLLM => rawLLM + (newLLM as string) + "\n")
    })

    socket.on("log", (newLog) => {
      setOvervierw(overview => overview + newLog + "\n")
    })

    socket.on("error", (newError) => {
      setOvervierw(overview => overview + "Error: " + newError + "\n")
    })


    socket.on("skill_message", newSkillMessage => {
      setSkillLog(skillLog => skillLog + newSkillMessage.skill + ": " + newSkillMessage.message + "\n")
    })

    // Possibly remove
    // socket.on("function_call", (newFunctionCall) => {
    //   const argStr = Object.keys(newFunctionCall["args"]).map(key => {
    //     return (key + "=\"" + newFunctionCall["args"][key]) + "\""
    //   }).join(", ")
    //   const newFunctionCallStr = newFunctionCall["function_name"] + "(" + argStr + ")"

    //   setFunctionCalls(functionCalls => [...functionCalls, newFunctionCallStr])
    // })

    socket.on("server_status", (status) => {
      setServerRunning(true)
      setSleeping(status["sleeping"])
      setRunSpeakFlag(status["speak_flag"])
      setResetDialogFlag(status["reset_dialog_flag"])
      setAvailableLLMModels(status["available_llm_models"] ?? [])
      setAvailableTranscribeModels(status["available_transcribe_models"] ?? [])
      setLLMSettings(status["llm_settings"] ?? {})
      setTranscribeSettings(status["transcribe_settings"] ?? {})
      setSkills(status["skills"] ?? [])

      setPromptPresets(status["prompt_presets"] ?? [])
      if (status["prompt_presets"].length > 0) {
        setPromptSetup(status["prompt_presets"][0].prompt)
      }

      if (status['run_llm'] === false) {
        // LLM not running
      }
    })

    socket.emit("request_status");

    // when component unmounts, disconnect
    return (() => {
      socket.disconnect()
    })
  }, [])

  const serverStatus = serverRunning ?
    sleeping ? "Sleeping" : runLLMFlag ? "Running" : "Listening, Not Automatically Responding"
    : "Disconnected from Server"

  return (
    <div>
      <AppBar position="static" sx={{ flexDirection: "row" }}   >
        <Typography marginLeft={"12px"}>Otto Bot</Typography>
        <Typography marginLeft={"auto"} marginRight={"12px"}>{serverStatus}</Typography>
      </AppBar>

      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleChange} aria-label="basic tabs example">
          <Tab label="System" {...a11yProps(0)} />
          <Tab label="Transcription" {...a11yProps(1)} />
          <Tab label="LLM" {...a11yProps(2)} />
          <Tab label="Skills" {...a11yProps(3)} />


        </Tabs>
      </Box>
      {overviewPanel()}
      {transcriptionPanel()}
      {llmPanel()}
      {skillPanel()}

    </div >


  )
}

export default System;
