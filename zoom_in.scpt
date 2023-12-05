tell application "Logic Pro X"
    activate
end tell

delay 0.5

tell application "System Events"
    tell process "Logic Pro X"
        if zoom_level > 0 then
            repeat zoom_level times
                keystroke "z"
            end repeat
        elseif zoom_level < 0 then
            repeat (-zoom_level) times
                keystroke "z" using {shift down}
            end repeat
        else
            keystroke "z"
        end if
    end tell
end tell
