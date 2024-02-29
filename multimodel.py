# Common libs
import argparse, os

# Model
from custom_model import MultiModel

# Main
if __name__ == '__main__':
    
    # Parse arguments
    desr = [
        'Multimodel for prediction of 7 mechanical properties of bcc alloys.',
        'Data directory and model.dump file need to be placed in the same path as this script.'
    ]
    
    parser = argparse.ArgumentParser(prog='MultiModel', description='\n'.join(desr), usage='%(prog)s [options]')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        required=True,
        help='provide filename consisting a column of chemical formulas'
    )
    
    args = parser.parse_args()
    print('Filename:', args.input)

    # Check file and model.dump
    file = args.input
    if not os.path.exists(file): 
        raise ValueError(f'{file} not found')
    
    if not os.path.exists('./data/model.dump'): 
        raise ValueError('"./data/model.dump" not found')

    # Load pretrained model
    multimodel = MultiModel.load_model('./data/model.dump')
    print('-'*79)
    print('Model loaded:')
    print('-'*79)
    print(multimodel)
    print('-'*79)

    # Predict & save results
    output_filename = f'{file}.results.csv'
    multimodel.predict_from_file(file=file).to_csv(output_filename, index=None)
    
    print('-'*79)
    print(f'Results saved to "{output_filename}"')