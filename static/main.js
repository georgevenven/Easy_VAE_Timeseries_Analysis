
function audio_file_input(){
    // set sonogram file input and segment file input to hidden
    document.getElementById("sonogram_file_input").style.display = "none";
    document.getElementById("segment_file_input").style.display = "none";

    // make the id audio_file_input visible
    document.getElementById("audio_file_input").style.display = "block";
    // make this id dir_location_form visible
    document.getElementById("dir_location_form").style.display = "block";
    
    // make upload input visible 
    document.getElementById("upload_input").style.display = "block";
}

function sonogram_file_input(){
    document.getElementById("audio_file_input").style.display = "none";
    document.getElementById("segment_file_input").style.display = "none";

    // make the id sonogram_file_input visible
    document.getElementById("sonogram_file_input").style.display = "block";
}

function segment_file_input(){
    document.getElementById("sonogram_file_input").style.display = "none";
    document.getElementById("audio_file_input").style.display = "none";
    // make the id segment_file_input visible
    document.getElementById("segment_file_input").style.display = "block";
}


function show_remove_empty_space_settings(){
    // switch from hidden to shown or vice versa id window_size and threshold
    var x = document.getElementById("window_size");
    var y = document.getElementById("threshold");
    if (x.style.display === "none") {
        x.style.display = "block";
        y.style.display = "block";
    }
    else {
        x.style.display = "none";
        y.style.display = "none";
    }
}


function start_training()
{
    // make the id stop_training and stop_training_no_save visible
    document.getElementById("stop_training").style.display = "block";
    document.getElementById("stop_training_no_save").style.display = "block";
    // make the id start_training invisible
    document.getElementById("start_training").style.display = "none";


}

function stop_training()
{
    // make the id stop_training and stop_training_no_save invisible
    document.getElementById("stop_training").style.display = "none";
    document.getElementById("stop_training_no_save").style.display = "none";
    // make the id start_training visible
    document.getElementById("start_training").style.display = "block";
}

function stop_training_no_save()
{
    // make the id stop_training and stop_training_no_save invisible
    document.getElementById("stop_training").style.display = "none";
    document.getElementById("stop_training_no_save").style.display = "none";
    // make the id start_training visible
    document.getElementById("start_training").style.display = "block";
}

function start_pre_processing_data()
{
    
}